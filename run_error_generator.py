import argparse
from error_generator import ErrorGeneratorForRREDv2 as ErrorGenerator

import json
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from health_multimodal.text.model import CXRBertModel
from health_multimodal.text.model import CXRBertTokenizer



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
    interpretive_error, finding_sent_label_path, impression_label_path,
    perceptive_error, bert_model, batch_size, device,
    ):

    ERROR_TYPE_LIST = get_error_type_list(
        factual_error,
        interpretive_error,
        perceptive_error
    )
    egv2 = ErrorGenerator()

    # load dataset
    egv2.load_original_samples(input_file)

    # get data for generating perceptual errors
    text_encoder = CXRBertModel.from_pretrained(
        bert_model, 
        revision="v1.1",
    )
    tokenizer = CXRBertTokenizer.from_pretrained(
        bert_model,
        revision='v1.1',
    )

    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        text_encoder = nn.DataParallel(text_encoder)
    text_encoder = text_encoder.to(device)

    text_encoder.eval()
    fds_emb_matrix, fds_index_map = egv2.get_embedding_matrix(
        egv2.original_finding_sents,
        text_encoder, tokenizer, device, batch_size
    )
    print('... LOAD FINDING SENTENCE EMBEDDING ...')
    imp_emb_matrix, imp_index_map = egv2.get_embedding_matrix(
        egv2.original_impressions,
        text_encoder, tokenizer, device, batch_size
    )
    print('... LOAD IMPRESSION EMBEDDING ...')

    # get data for generating interpretive errors
    imp_fd_map = egv2.get_imp_fd_pair_map()
    print('... LOAD IMPRESSION FINDING MAP ...')
    imp_label_df, fds_label_df = egv2.get_chexpert_label(
        impression_label_path, finding_sent_label_path,
        class_group_info=None
        )
    print('... LOAD CHEXPERT LABLER WITH PANDAS.DATAFRAME ...')

    error_subtype_counter = {}
    with open(output_file, encoding= "utf-8",mode='w+') as fw:
        for idx, sample in enumerate(egv2.original_samples):
            if (idx+1) % 500 == 0:
                print(f'{idx+1}/{len(egv2.original_samples)} finish ...')

            error_type = np.random.choice(ERROR_TYPE_LIST, 1, replace=False).item()
            #print(error_type)

            if error_type == 'F':
                error_findings_dict = egv2.generate_factual_error(
                    sample['Findings'],
                    n_prob=0.25, u_prob=0.25, l_prob=0.1
                )

            elif error_type == 'P':
                error_findings_dict = egv2.generate_perceptual_error(
                    fds_emb_matrix, fds_index_map, 
                    imp_emb_matrix, imp_index_map,
                    sample['Findings'],
                    sample['Impression'],
                    swap_prob=0.5
                )

            elif error_type == 'I':
                error_findings_dict = egv2.generate_interpretive_error(
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
    parser.add_argument("--input", type=str, default="/home/data/mimic-cxr-jpg/2.0.0/rred/frontal_test.jsonl", help="input file of unaugmented data")

    parser.add_argument("--output", type=str, default="/home/workspace/rred_v2_frontal_test_error.jsonl", help="output file of unaugmented data")
#    parser.add_argument("--output", type=str, default="/home/data/mimic-cxr-jpg/2.0.0/rred/error_baseline_Mixed_FPI_v0.1/frontal_test_error.jsonl", help="output file of unaugmented data")

    parser.add_argument("--factual_error", type=bool, choices=[True, False], default=True, help="True if we want to generate factual errors")

    parser.add_argument("--interpretive_error", type=bool, choices=[True, False], default=True, help="True if we want to generate interpretive errors")
    parser.add_argument("--finding_sent_label_path", default="/home/data/CheXpert_labeler_result/labeled_chexpert_finding_sent.csv", type=str)
    parser.add_argument("--impression_label_path", default="/home/data/CheXpert_labeler_result/labeled_chexpert_impression.csv", type=str)

    parser.add_argument("--perceptive_error", type=bool, choices=[True, False], default=True)
    parser.add_argument("--bert_model", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized",
                        choices=["bert-base-uncased, emilyalsentzer/Bio_ClinicalBERT", "xlm-roberta-base", '/home/workspace/Multi-modality-Self-supervision/GatorTron', 'microsoft/BiomedVLP-CXR-BERT-specialized'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default='cuda')
 #   parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    print("="*30, "ARGUMENT INFO", "="*30)
    for k, v in args.__dict__.items():
        print(f'{k}: {v}')
    print("="*30, "ARGUMENT INFO", "="*30)

    generate_error(
        args.input, args.output, 
        args.factual_error,
        args.interpretive_error, args.finding_sent_label_path, args.impression_label_path,
        args.perceptive_error, args.bert_model, args.batch_size, args.device,
    )

    
if __name__ == "__main__":
    
    import time
    start = time.time()
    main()
    print("time :", time.time() - start)
