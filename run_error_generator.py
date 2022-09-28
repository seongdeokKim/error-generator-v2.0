import argparse
from error_generator import ErrorGeneratorForRREDv2 as ErrorGenerator

import json
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# CXRbert
#from models.configuration_cxrbert import CXRBertConfig
#from models.modelling_cxrbert import CXRBertModel
#from models.configuration_cxrbert import CXRBertTokenizer
#from health_multimodal.text.model import CXRBertModel
#from health_multimodal.text.model import CXRBertTokenizer

# CheXbert
from models.bert_labeler import bert_labeler
from transformers import BertTokenizer
from constants import *

def get_error_type_list(
    factual_error,
    interpretive_error,
    perceptive_error
    ):

    ERROR_TYPE_LIST = []
    if factual_error == True:
        ERROR_TYPE_LIST.append('F')
    if interpretive_error == True:
        ERROR_TYPE_LIST.append('I')
    if perceptive_error == True:
        ERROR_TYPE_LIST.append('P') 

    return ERROR_TYPE_LIST

def generate_error(
    input_file, output_file,
    factual_error,
    perceptive_error, embedding_cache_path, chexbert_model_path, batch_size, device_mode,
    interpretive_error, 
    finding_sent_label_path, finding_label_path, impression_label_path,
    ):

    ERROR_TYPE_LIST = get_error_type_list(
        factual_error,
        interpretive_error,
        perceptive_error
    )
    eg_v2 = ErrorGenerator()
    eg_v2.chexbert_categories = CATEGORIES

    # load dataset
    eg_v2.load_original_samples(input_file)

    chexbert_model = bert_labeler()
    chexbert_tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased'
    )

    device = torch.device("cuda" if torch.cuda.is_available() and device_mode == "cuda" else "cpu")
    if torch.cuda.device_count() > 0 and device_mode == "cuda":
        print("Using", torch.cuda.device_count(), "GPUs!")

        chexbert_model = nn.DataParallel(chexbert_model)
        chexbert_model = chexbert_model.to(device)
        print(f"...... LOADING CheXbert MODEL's WEIGHT FROM THE CHECKPOINT[{chexbert_model_path}] ......")
        chex_checkpoint = torch.load(chexbert_model_path)
        chexbert_model.load_state_dict(chex_checkpoint['model_state_dict'   ])
    else:
        print("Using CPU!")


        print("...... LOADING CheXbert MODEL ......")
        from collections import OrderedDict
        chexbert_model = chexbert_model.to(device)
        print(f"...... LOADING CheXbert MODEL's WEIGHT FROM THE CHECKPOINT[{chexbert_model_path}] ......")
        checkpoint = torch.load(chexbert_model_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        chexbert_model.load_state_dict(new_state_dict, strict=True)
#        raise NotImplementedError("Use GPU")

    chexbert_model.eval()

    print('...... LOADING TEXT EMBEDDING ......')
    embedding_matrix, text_index_map = eg_v2._get_embedding_matrix(embedding_cache_path)

    # get data for generating interpretive errors
    imp_fd_map = eg_v2.get_imp_fd_pair_map()
    print('...... LOADING IMPRESSION FINDING MAP ......')

    imp_label_df, fd_label_df, fds_label_df = eg_v2.get_chexpert_label(
        impression_label_path, finding_label_path, finding_sent_label_path,
        class_group_info=None
        )
    print('...... LOADING CHEXPERT LABLER WITH PANDAS.DATAFRAME ......')

    fw_test = open('/home/workspace/new_p_error_test.txt', 'w', encoding='utf-8')
    error_subtype_counter = {}
    with open(output_file, encoding= "utf-8",mode='w+') as fw:
        for idx, sample in enumerate(eg_v2.original_samples):
            if sample['Findings'].strip() == '' or sample['Findings'] is None:
                continue
            if sample['Impression'].strip() == '' or sample['Impression'] is None:
                continue

            if (idx+1) % 500 == 0:
                print(f'... {idx+1}/{len(eg_v2.original_samples)} finish ...')

#            error_type = np.random.choice(ERROR_TYPE_LIST, 1, replace=False).item()
            error_type = 'P'

            if error_type == 'F':
                error_findings_dict = eg_v2.generate_factual_error(
                    sample['Findings'],
                    n_prob=0.25, u_prob=0.25, l_prob=0.1
                )

            elif error_type == 'P':
                # error_findings_dict = eg_v2.generate_perceptual_error(
                #     fds_emb_matrix, fds_index_map, 
                #     imp_emb_matrix, imp_index_map,
                #     sample['Findings'], sample['Impression'],
                #     chexbert_model, chexbert_tokenizer, batch_size, device,
                #     fd_label_df,
                #     swap_prob=0.5
                # )
                error_findings_dict = eg_v2._generate_perceptual_error(
                    embedding_matrix, text_index_map, 
                    sample['Findings'], sample['Impression'],
                    chexbert_model, chexbert_tokenizer, batch_size, device,
                    swap_prob=0.5
                )

            elif error_type == 'I':
                error_findings_dict = eg_v2.generate_interpretive_error(
                    sample['Findings'],
                    sample['Impression'],
                    imp_fd_map, 
                    imp_label_df, fds_label_df,
                    sf_prob=1, ss_prob=1, as_prob=1
                )


            for error_subtype, error_finding_list in error_findings_dict.items():
                if error_subtype == 'swapped_sentences':
                    continue
                if error_subtype == 'added_sentences':
                    continue

                ##################################################
                ##################################################
                if error_subtype == 'under_reading' or error_subtype == 'satisfaction_of_search':
                    for error_finding in error_finding_list:
                        original_label = "\t".join(list(map(str, error_findings_dict['original'][0][1])))
                        error_finding_label = "\t".join(list(map(str, error_finding[1])))

                        fw_test.write('##{}\t{}\t{}\n##{}\t{}\t{}\n\n'.format(
                            str(idx), error_findings_dict['original'][0][0], original_label,
                            str(idx) + "_" + error_subtype, error_finding[0], error_finding_label
                        ))
                    continue
                ##################################################
                ##################################################
                for error_finding in error_finding_list:
                    error_sample = dict(sample)

                    error_sample['Findings'] = error_finding
                    error_sample['error_subtype'] = error_subtype
                    error_sample['original_Findings'] = sample['Findings']
                            
                    json_record = json.dumps(error_sample, ensure_ascii=False)
                    fw.write(json_record+"\n")

                    if error_subtype not in error_subtype_counter:
                        error_subtype_counter[error_subtype] = 0
                    error_subtype_counter[error_subtype] += 1
        print(error_subtype_counter)

def main():
    parser = argparse.ArgumentParser()
#    parser.add_argument("--input", type=str, help="input file of unaugmented data")
    parser.add_argument("--input", type=str, default="/home/data/mimic-cxr-jpg/2.0.0/rred/frontal_val.jsonl", help="input file of unaugmented data")

    parser.add_argument("--output", type=str, default="/home/workspace/rred_v2_frontal__test_error.jsonl", help="output file of unaugmented data")
#    parser.add_argument("--output", type=str, default="/home/data/mimic-cxr-jpg/2.0.0/rred/error_baseline_Mixed_FPI_v0.1/frontal_test_error.jsonl", help="output file of unaugmented data")

    parser.add_argument("--factual_error", type=bool, choices=[True, False], default=False, help="True if we want to generate factual errors")

    parser.add_argument("--perceptive_error", type=bool, choices=[True, False], default=True)
    parser.add_argument("--embedding_cache_path", type=str, default="/home/data/cxr_bert/embedding_cache_path.pkl", help="embeddings of text using CXR-BERT")
    parser.add_argument("--chexbert_model_path", type=str, default="/home/workspace/models/chexbert.pth")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--device", type=str, default='cuda')
#    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--interpretive_error", type=bool, choices=[True, False], default=False, help="True if we want to generate interpretive errors")
    parser.add_argument("--finding_sent_label_path", default="/home/data/CheXpert_labeler_result/labeled_chexpert_finding_sent.csv", type=str)
    parser.add_argument("--finding_label_path", default="/home/data/CheXpert_labeler_result/labeled_chexpert_finding.csv", type=str)
    parser.add_argument("--impression_label_path", default="/home/data/CheXpert_labeler_result/labeled_chexpert_impression.csv", type=str)

    args = parser.parse_args()

    print("="*30, "ARGUMENT INFO", "="*30)
    for k, v in args.__dict__.items():
        print(f'{k}: {v}')
    print("="*30, "ARGUMENT INFO", "="*30)

    generate_error(
        args.input, args.output, 
        args.factual_error,
        args.perceptive_error, args.embedding_cache_path, args.chexbert_model_path, args.batch_size, args.device,
        args.interpretive_error, 
        args.finding_sent_label_path, args.finding_label_path, args.impression_label_path,
    )

    
if __name__ == "__main__":
    
    import time
    start = time.time()
    main()
    print("time :", time.time() - start)
