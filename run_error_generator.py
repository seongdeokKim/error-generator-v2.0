import argparse
from error_generator import ErrorGeneratorForRREDv2 as ErrorGenerator

import json
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# CheXbert
from models.bert_labeler import bert_labeler
from transformers import BertTokenizer
from constants import *

def get_error_types_with_prob(
    factual_miss=0.09,
    under_reading=0.49,
    satisfaction_of_search=0.28,
    complacency=0.01,
    faulty_reasoning=0.12,
    random_swap=0.01,
    ):

    ERROR_TYPES = []
    ERROR_TYPE_PROB = []

    if factual_miss > 0:
        ERROR_TYPES.append('FM')
        ERROR_TYPE_PROB.append(factual_miss)
    if under_reading > 0:
        ERROR_TYPES.append('UR')
        ERROR_TYPE_PROB.append(under_reading)
    if satisfaction_of_search > 0:
        ERROR_TYPES.append('SoS')
        ERROR_TYPE_PROB.append(satisfaction_of_search)
    if complacency > 0:
        ERROR_TYPES.append('C')
        ERROR_TYPE_PROB.append(complacency)
    if faulty_reasoning > 0:
        ERROR_TYPES.append('FR')
        ERROR_TYPE_PROB.append(faulty_reasoning)
    if random_swap > 0:
        ERROR_TYPES.append('RS')
        ERROR_TYPE_PROB.append(random_swap)

    return ERROR_TYPES, ERROR_TYPE_PROB


def generate_error(
    input_file, output_file,
    embedding_cache_path, 
    chexbert_model_path, batch_size, device_mode,
    chexpert_label_of_finding_sent,
    max_epoch
    ):

    ERROR_TYPES, _ = get_error_types_with_prob()

    eg_v2 = ErrorGenerator()

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

    print('...... LOADING FINDING & IMPRESSION EMBEDDING using CXR-BERT......')
    embedding_matrix, text_index_map = eg_v2.get_embedding_matrix(embedding_cache_path)

    # get data for generating interpretive errors
    print('...... LOADING FINDING LABEL using CheXbert ......')
    finding_label_df = eg_v2.get_chexbert_label_of_finding(
        chexbert_model, chexbert_tokenizer, batch_size, device,
        eg_v2.original_findings
    )

    print('...... LOADING CHEXPERT LABLER WITH PANDAS.DATAFRAME ......')
    fd_sent_label_df = eg_v2.get_chexpert_label_of_finding_sents(chexpert_label_of_finding_sent)

    fw_test = open('/home/workspace/error_generator_result.txt', 'w', encoding='utf-8')
    error_subtype_counter = {}
    for epoch in range(int(max_epoch)):
        epoch_start = time.time()
        with open(output_file.replace('.jsonl', f'_e{epoch+1}.jsonl'), encoding= "utf-8",mode='w+') as fw:
            for idx, sample in enumerate(eg_v2.original_samples):
                if sample['Findings'].strip() == '' or sample['Findings'] is None:
                    continue

                if sample['Impression'].strip() == '' or sample['Impression'] is None:
                    continue

                if (idx+1) % 100 == 0:
                    print(f'... EPOCH {epoch+1} ::: {idx+1}/{len(eg_v2.original_samples)} finish ...')

                current_error_type = np.random.choice(ERROR_TYPES, 1, replace=False).item()
                # current_error_type = np.random.choice(ERROR_TYPES, 1, p=ERROR_TYPE_PROB,replace=False).item()
                # current_error_type = 'FM'
                #current_start = time.time()

                if current_error_type == 'FM':
                    try:
                        output = eg_v2.generate_factual_miss(
                            sample['Findings'],
                            n_prob=0.45, u_prob=0.45, l_prob=0.1
                        )
                    except Exception as e:
                        print(current_error_type, e)
                        continue
                elif current_error_type == 'UR':
                    try:
                        output = eg_v2.generate_under_reading(
                            embedding_matrix, text_index_map, finding_label_df,
                            sample['Findings'], sample['Impression'],
                            chexbert_model, chexbert_tokenizer, batch_size, device,
                            del_prob=0.3
                        )
                    except Exception as e:
                        print(current_error_type, e)
                        continue
                elif current_error_type == 'SoS':
                    try:
                        output = eg_v2.generate_satisfaction_of_search(
                            embedding_matrix, text_index_map, finding_label_df,
                            sample['Findings'], sample['Impression'],
                            chexbert_model, chexbert_tokenizer, batch_size, device,
                            del_prob=0.3
                        )
                    except Exception as e:
                        print(current_error_type, e)
                        continue        
                elif current_error_type == 'C':
                    try:
                        output = eg_v2.generate_complacency(
                            sample['Findings'],
                            finding_label_df, fd_sent_label_df,
                            chexbert_model, chexbert_tokenizer, batch_size, device,
                            num_of_insert_sent=1
                        )
                    except Exception as e:
                        print(current_error_type, e)
                        continue
                elif current_error_type == 'FR':
                    try:
                        output = eg_v2.generate_faulty_reasoning(
                            sample['Findings'], 
                            finding_label_df, fd_sent_label_df,
                            chexbert_model, chexbert_tokenizer, batch_size, device,
                            fr1_prob=.5, fr2_prob=.5, swap_prob=.5
                        )
                    except Exception as e:
                        print(current_error_type, e)
                        continue
                elif current_error_type == 'RS':
                    try:
                        output = eg_v2.generate_random_swap(
                            sample['Findings'],
                            finding_label_df
                        )
                    except Exception as e:
                        print(current_error_type, e)
                        continue

                for error_subtype, error_findings in output.items():
                    for error_finding in error_findings:
                        if error_subtype not in error_subtype_counter:
                            error_subtype_counter[error_subtype] = 0
                        error_subtype_counter[error_subtype] += 1

                        if error_subtype in ['numerical_error', 'unit_error', 'laterality_error']:
                            error_sample = dict(sample)

                            error_sample['Findings'] = error_finding
                            error_sample['error_subtype'] = error_subtype
                            error_sample['original_Findings'] = sample['Findings']
                                    
                            json_record = json.dumps(error_sample, ensure_ascii=False)
                            fw.write(json_record+"\n")

                            fw_test.write('##{}_org\t{}\t{}\n##{}_error\t{}\t{}\n'.format(
                                str(idx), error_subtype, error_sample['original_Findings'],
                                str(idx), error_subtype, error_sample['Findings']
                            ))

                        else:
                            
                            fd_features = finding_label_df[
                                finding_label_df[REPORTS] == sample['Findings']
                            ].values[0][1:]
                            _fd_features = fd_features[:-1]
                            original_labels = "\t".join(list(map(str, _fd_features)))
                            error_labels = "\t".join(list(map(str, error_finding[1])))

                            error_sample = dict(sample)

                            error_sample['error_subtype'] = error_subtype
                            error_sample['Findings'] = error_finding[0]
                            error_sample['labels'] = error_labels
                            error_sample['original_Findings'] = sample['Findings']
                            error_sample['original_labels'] = original_labels

                            json_record = json.dumps(error_sample, ensure_ascii=False)
                            fw.write(json_record+"\n")

                            fw_test.write('##{}_org\t{}\t{}\t{}\n##{}_error\t{}\t{}\t{}\n'.format(
                                str(idx), error_subtype, error_sample['original_Findings'], error_sample['original_labels'],
                                str(idx), error_subtype, error_sample['Findings'], error_sample['labels']
                            ))

                #print(f"{current_error_type} error, {str(time.time()-current_start)[:5]}")
        
        print(f"{str(time.time()-epoch_start)[:5]}")

    print(error_subtype_counter)

def main():
    parser = argparse.ArgumentParser()
#    parser.add_argument("--input", type=str, help="input file of unaugmented data")
    parser.add_argument("--input", type=str, default="/home/data/mimic-cxr-jpg/2.0.0/rred/frontal_test.jsonl", help="input file of unaugmented data")

    # parser.add_argument("--output", type=str, default="/home/workspace/rred_v2_frontal_val_error.jsonl", help="output file of unaugmented data")
    parser.add_argument("--output", type=str, default="/home/data/mimic-cxr-jpg/2.0.0/rred/error_baseline_Mixed_FPI_v0.4/frontal_test_error.jsonl", help="output file of unaugmented data")

    parser.add_argument("--cxr_bert_embedding_cache_path", type=str, default="/home/data/text_embedding_of_cxr-bert/text_embedding_path.pkl", help="embeddings of text using CXR-BERT")
    parser.add_argument("--chexbert_model_path", type=str, default="/home/workspace/models/chexbert.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default='cuda')
    # parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--chexpert_label_of_finding_sent", default="/home/data/CheXpert_labeler_result/labeled_chexpert_finding_sent.csv", type=str)

    parser.add_argument("--max_epoch", default=10, type=int)

    args = parser.parse_args()

    print("="*30, "ARGUMENT INFO", "="*30)
    for k, v in args.__dict__.items():
        print(f'{k}: {v}')
    print("="*30, "ARGUMENT INFO", "="*30)

    generate_error(
        args.input, args.output, 
        args.cxr_bert_embedding_cache_path, 
        args.chexbert_model_path, args.batch_size, args.device,
        args.chexpert_label_of_finding_sent,
        args.max_epoch
    )

    
if __name__ == "__main__":
    
    import time
    start = time.time()
    main()
    print("time :", time.time() - start)