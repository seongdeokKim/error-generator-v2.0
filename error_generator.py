from doctest import script_from_examples
import numpy as np
from numpy.linalg import norm
import random

import torch
import pandas as pd
import random
import re
import json
import nltk
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')

class ErrorGeneratorForRREDv2:
    def __init__(self):

        # error 유형: 
        # 1) factual error -> 0~1개
        # 2) perceptual error -> 1개
        # 3) interpretive error -> 1개
        # whole process(pipeline)
        # A%/B%/C% 확률로 특정 error generate 함수 실행하여 final 1개 error finding 생성
        # 1)의 경우 없으면 에러 처리 및 제외

        self.original_samples = [] # list of dict. each Dict represents each sample(radiology report)
        self.original_findings = []
        self.original_finding_sents = []
        self.original_impressions = []

        self.chexpert_label_group = {
            "GROUP1": ["Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Lung Lesion", "Atelectasis"],
            "GROUP2": ["Enlarged Cardiomediastinum", "Cardiomegaly"],
            "OTHERS": ["Reports", "No Finding", "Support Devices", "Fracture", "Pleural Other", "Pleural Effusion", "Pneumothorax"]
        }

    def load_original_samples(self, input_path):
        self.original_samples = [json.loads(l) for l in open(input_path)]
        for sample in self.original_samples:
            self.original_findings.append(sample['Findings'])
            self.original_impressions.append(sample['Impression'])
            self.original_finding_sents += sent_tokenize(sample['Findings'])

    def get_imp_fd_pair_map(self):
        train_data_path = "/home/data/mimic-cxr-jpg/2.0.0/rred/frontal_train.jsonl"
        valid_data_path = "/home/data/mimic-cxr-jpg/2.0.0/rred/frontal_val.jsonl"
        test_data_path = "/home/data/mimic-cxr-jpg/2.0.0/rred/frontal_test.jsonl"

        input_path_list = [
            train_data_path, valid_data_path, test_data_path
        ]

        imp_fd_map = {}
        for input_path in input_path_list:
            samples = [json.loads(l) for l in open(input_path)]
            for sample in samples:
                imp, fd = sample['Impression'], sample['Findings']
                if imp not in imp_fd_map:
                    imp_fd_map[imp] = []
                imp_fd_map[imp].append(fd)

        for imp in imp_fd_map.keys():
            imp_fd_map[imp] = list(set(imp_fd_map[imp]))
        
        return imp_fd_map

    def get_chexpert_label(
        self, impression_label_path, finding_sent_label_path, 
        class_group_info=None
        ):

        imp_label_df = pd.read_csv(impression_label_path)
        fd_sent_label_df = pd.read_csv(finding_sent_label_path)

        if class_group_info is None:
            pass    

        return imp_label_df, fd_sent_label_df

    ########################################################################
    ########################################################################
    #############    FACTUAL ERROR
    ########################################################################
    ########################################################################

    def generate_factual_error(
        self, document: str,
        n_prob=0.25, u_prob=0.25, l_prob=0.1
        ):
        # number_error & unit_error : max 1개, lateral_error: max N개
        # minimum 0 ~ maximum N+2개
        # 이중에서 random으로 1개의 error만 반환
        # error 0개일 경우 type = 'No' => 별도의 파일로 저장(추후 참조용)
        
        # candidates = []

        # candidates += self.generate_number_error(document)
        # candidates += self.generate_unit_error(document)
        # candidates += self.generate_laterality_error(document)

        # if len(candidates) == 0:
        #     return None
        
        # final_error = np.random.choice(candidates, 1, replace=False).item()
        # return final_error

        generated_errors = {
            'numerical_errors': [],
            'unit_errors': [],
            'laterality_errors': [],
        }

        numerical_errors = self._get_numerical_error(document)
        if len(numerical_errors) > 0:
            if np.random.choice([True, False], 1, p=[n_prob, 1-n_prob]).item():
                # shuffling numerical_errors
                random.shuffle(numerical_errors)
                generated_errors['numerical_errors'] += [numerical_errors[-1]]

        unit_errors = self._get_unit_error(document)
        if len(unit_errors) > 0:
            if np.random.choice([True, False], 1, p=[u_prob, 1-u_prob]).item():
                random.shuffle(unit_errors)
                generated_errors['unit_errors'] += [unit_errors[-1]]
        
        laterality_errors = self._get_laterality_error(document)
        if len(laterality_errors) > 0:
            random.shuffle(laterality_errors)
            if len(numerical_errors) == 0 and len(unit_errors) == 0:
                generated_errors['laterality_errors'] += [laterality_errors[-1]]
            else:
                if np.random.choice([True, False], 1, p=[l_prob, 1-l_prob]).item():
                    generated_errors['laterality_errors'] += [laterality_errors[-1]]

        return generated_errors

    def _get_numerical_error(self, doc: str):

        numerical_errors = []
        
        numeric_units = re.findall(r'(?:\d+(?:\.\d+)?(?::?\d+)?)(?:\s*[a|p|c]?\.?m\.?m?)', doc)
        numeric_units = [nu for nu in numeric_units if "cm" in nu or "mm" in nu]
        numeric_units = [nu if not nu.endswith('.') else nu[:-1] for nu in numeric_units]

        numeric_units = list(set(numeric_units))

        if len(numeric_units) == 0:
            return []

        generated = []
        for nu in numeric_units:
            if "mm" in nu:
                numeric = nu.replace("mm", "").strip()
            elif "cm" in nu:
                numeric = nu.replace("cm", "").strip()

            random_value = np.random.choice(
                [10, 100], 1, replace=False, p=[0.5, 0.5]
            ).item()
            candidate_numeric1 = float(numeric) * random_value
            candidate_numeric2 = float(numeric) / random_value
            
            final_numeric = np.random.choice(
                [candidate_numeric1, candidate_numeric2], 1, replace=False, p=[0.5, 0.5]
            ).item()
            
            if len(str(final_numeric)) > 7:
                final_numeric = round(final_numeric)

            generated.append(
                nu.replace(str(numeric), str(final_numeric))
            )

        for i in range(len(numeric_units)):
#            print(numeric_units[i], "=====>>>>>",generated[i])
#            print(doc.replace(numeric_units[i], generated[i]))
#            print("="*100)

            numerical_errors.append(
                doc.replace(numeric_units[i], generated[i])
            )

        return numerical_errors

    def _get_unit_error(self, doc: str):
        unit_errors = []

        numeric_units = re.findall(r'(?:\d+(?:\.\d+)?(?::?\d+)?)(?:\s*[a|p|c]?\.?m\.?m?)', doc)
        numeric_units = [nu for nu in numeric_units if "cm" in nu or "mm" in nu]
        numeric_units = [nu if not nu.endswith('.') else nu[:-1] for nu in numeric_units]

        numeric_units = list(set(numeric_units))

        if len(numeric_units) == 0:
            return []

        #random.shuffle(numeric_units)
        generated = []
        for nu in numeric_units:
            if "mm" in nu: 
                generated.append(
                    nu.replace("mm", "cm")
                )
            elif "cm" in nu:
                generated.append(
                    nu.replace("cm", "mm")
                )

        new_doc = doc
        for i in range(len(numeric_units)):
            unit_errors.append(
                new_doc.replace(numeric_units[i], generated[i])
            )

        return unit_errors

    def _get_laterality_error(self, doc: str):

        laterality_dict = {
            "left_right" : {
                " right " : " left ",
                " left " : " right ",
                " Right " : " Left ",
                " Left " : " Right ",
            },
            "upper_lower" : {
                " upper " : " lower ",
                " lower " : " upper ",
                " Upper " : " Lower ",
                " Lower " : " Upper "
            },
#            "high_low" : {
#                " high " : " low ",
#                " low " : " high ",
#                " High " : " Low ",
#                " Low " : " High "
#            },
            "lt_rt" : {
                " lt " : " rt ",
                " rt " : " lt ",
                " Rt " : " Lt ",
                " Lt " : " Rt ",
                " lt." : " rt.",
                " rt." : " lt.",
                " Rt." : " Lt.",
                " Lt." : " Rt.",

            }
        }

        pair_dicts = []
        for key_pair in laterality_dict.keys():
            key_pair_list = list(map(lambda key: f' {key} ', key_pair.split("_")))
            key1, key2 = key_pair_list[0], key_pair_list[1]
            if key1 in doc.lower() or key2 in doc.lower():
                pair_dicts.append(laterality_dict[key_pair])

        if len(pair_dicts) == 0:
            return []

        laterality_errors = []
        for i in range(len(pair_dicts)):
            pair_dict = pair_dicts[i]
            regex = re.compile("(%s)" % "|".join(map(re.escape, pair_dict.keys())))
            laterality_error = regex.sub(
                lambda mo: pair_dict[mo.string[mo.start():mo.end()]], 
                doc
            )
            laterality_errors.append(laterality_error)

        return laterality_errors
        
    def generate_perceptual_error(
        self, fs_emb_matrix, fs_index_map, 
        imp_emb_matrix, imp_index_map,
        finding: str, impression: str, swap_prob
        ):

        def cos_sim(A, B):
            return np.dot(A, B)/(norm(A)*norm(B))

        f_sents = sent_tokenize(finding)
        sim_dist = []
        for f_sent in f_sents:
            sim_dist.append(
                cos_sim(imp_emb_matrix[imp_index_map[impression]], 
                        fs_emb_matrix[fs_index_map[f_sent]])
            )

        # get indexes of candidate sentences which should be eliminated
        eliminated_idxs = []
        _sim_dist = list(sim_dist)
        while len(eliminated_idxs) < max(1, round(len(sim_dist)*0.5)):
            tmp = max(_sim_dist)
            eliminated_idxs.append(
                sim_dist.index(tmp)
            )
            _sim_dist.remove(tmp)

        # eliminate or swap some sentences corresponding to eliminated indexes
        error_log = []
        for _ in range(len(eliminated_idxs)):
            error_log.append(
                np.random.choice([True, False], 1, p=[swap_prob, 1-swap_prob]).item()
            )

        # if set minimum error rate for the case which error rate is lower than 25%
        import math
        while error_log.count(True) < math.ceil(len(sim_dist)*0.25):
            # # of sents     :  3  4  5  6  7  8  9  10
            # max # of error :  1  1  2  2  2  2  3   3 
            random_idx = random.randrange(len(error_log))
            error_log[random_idx] = True

        current_log_idx = -1
        final_finding = []
        for idx in range(len(f_sents)):
            if idx in eliminated_idxs:
                current_log_idx += 1
                if error_log[current_log_idx]: 
                    #final_finding.append('[ELIMINATE]')
                    pass
                else:
                    final_finding.append(f_sents[idx])
            else:
                final_finding.append(f_sents[idx])

        final_finding = " ".join(final_finding)

        return {
            'perceptual_error': [final_finding]
        }

    def get_embedding_matrix(
        self, original_samples, 
        text_encoder, tokenizer, device, batch_size
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from health_multimodal.text.model import CXRBertModel
        from health_multimodal.text.model import CXRBertTokenizer

        with torch.no_grad():
            samples = list(set(original_samples))
            index_map = {sample: i for i, sample in enumerate(samples)}
            embedding_matrix = []
            for idx in range(0, len(samples), batch_size):
                mini_batch = tokenizer(
                    samples[idx:idx + batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = torch.tensor(mini_batch['input_ids'], dtype=torch.long)
                attention_mask = torch.tensor(mini_batch['attention_mask'], dtype=torch.long)
                
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                last_hidden_states = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )['last_hidden_state']
                # last_hidden_states = (batch_size, max_seq_len, hidden_size)
                cls_embs = last_hidden_states[:, 0, :].cpu().detach().numpy()
                # cls_embs = (batch_size, hidden_size)
                embedding_matrix.append(cls_embs)

                print(f'... {idx+batch_size}/{len(samples)} ...')

            # Concatenate the mini-batch wise result
            embedding_matrix = np.concatenate(embedding_matrix, axis=0) 
            # embedding_matrix = (len(samples), hidden_size)

        return embedding_matrix, index_map

#    def get_swaped_sentence(self, sent, doc, docs):
#        # 동일 Finding 내의 문장 X
#        # n개의 candidate 뽑고 개별적인 cosine_similarity 계산 후 가장 낮은 랭킹의 문장 반환
#        while True:
#            random_idx = random.randrange(len(docs))
#            random_sent = docs[random_idx]
#            if random_sent != sent:
#
#                return random_sent

    def generate_interpretive_error(
        self, finding: str, impression: str,
        imp_fd_map: dict, 
        imp_label_df, fd_sent_label_df,
        sf_prob: float , ss_prob: float, as_prob: float
    ):
        # 서로 다른 finding이 같은 impression을 갖는 경우 존재
        # chexpert_labeler 결과, impression만 존재하며 finding은 없음 -> 추후 mapping 작업도 필요

        generated_errors = {
            'swapped_findings': [],
            'swapped_sentences': [],
            'added_sentences': [],
        }

        swapped_findings = self._get_swap_finding(
            impression,
            imp_fd_map, 
            imp_label_df
        )
        if len(swapped_findings) > 0:
            if np.random.choice([True, False], 1, p=[sf_prob, 1-sf_prob]).item():
                # shuffling numerical_errors
                random.shuffle(swapped_findings)
                generated_errors['swapped_findings'] += [swapped_findings[-1]]

        swapped_sentences = self._get_swap_sentence(
            finding,
            fd_sent_label_df
        )

        added_sentences = self._get_add_sentence(
            finding,
            fd_sent_label_df
            )

        return generated_errors

    def _get_swap_finding(self, imp: str, imp_fd_map, imp_label_df):

        cls_list = imp_label_df.keys()

        imp_features = imp_label_df[
            imp_label_df['Reports'] == imp
        ].values[0]
        # 1. pos + neg -> 합집합O, 교집합X
        # 2. 50%/50%: ~fuzzy(=random) OR fuzzy func. 기준 유사도 높은 candidates(but label 상이) 추출

        pos_cls_idxs = np.where(imp_features == float(1))[0]
        pos_clss = [cls_list[idx] for idx in pos_cls_idxs]
        #pos_clss = list(map(lambda x: cls_list[x], pos_cls_idxs))

        neg_cls_idxs = np.where(imp_features == float(0))[0]
        neg_clss = [cls_list[idx] for idx in neg_cls_idxs]
        #neg_clss = list(map(lambda x: cls_list[x], neg_cls_idxs))

        pos_opp_imp_idxes = []
        for pos_cls in pos_clss:
            # if pos_cls in self.chexpert_label_group["GROUP1"]:
            #     imp_label_df = imp_label_df[
            #         self.chexpert_label_group["GROUP2"] + self.chexpert_label_group["OTHERS"]
            #     ]
            # elif pos_cls in self.chexpert_label_group["GROUP2"]:
            #     imp_label_df = imp_label_df[
            #         self.chexpert_label_group["GROUP1"] + self.chexpert_label_group["OTHERS"]
            #     ]            
            curr_pos_opp_imp_idxes = imp_label_df[
                imp_label_df[pos_cls] == float(0)
            ].index
            pos_opp_imp_idxes += list(curr_pos_opp_imp_idxes)

        neg_opp_imp_idxes = []
        for neg_cls in neg_clss:
            curr_neg_opp_imp_idxes = imp_label_df[
                imp_label_df[neg_cls] == float(1)
            ].index
            neg_opp_imp_idxes += list(curr_neg_opp_imp_idxes)


        opp_imp_idxes = pos_opp_imp_idxes + neg_opp_imp_idxes
        opp_imp_idxes = list(set(opp_imp_idxes))
        opp_imp_idxes.sort(reverse=False)

        opp_imps = list(imp_label_df.iloc[opp_imp_idxes, 0])

        if len(opp_imps) < 1:
            return []

        select_method = np.random.choice(['random', 'similarity'], 1, p=[0.5, 0.5]).item()
        #select_method = np.random.choice(['random', 'similarity'], 1, p=[0.0, 1.0]).item()
        #select_method = np.random.choice(['random', 'similarity'], 1, p=[1.0, 0.0]).item()

        swapped_findings = []
        if select_method == 'random':
            for opp_imp in opp_imps:
                swapped_findings += imp_fd_map[opp_imp]

            #random.shuffle(swapped_findings)
            #return swapped_findings[-1]

        elif select_method == 'similarity':
            opp_imp_features_list = imp_label_df.iloc[opp_imp_idxes, :].values
            _opp_imp_features_list = [
                list(map(str, opp_imp_features[1:])) for opp_imp_features in opp_imp_features_list
            ]

            _imp_features = list(map(str, imp_features[1:]))

            sim_list = []
            for _opp_imp_features in _opp_imp_features_list:
                match_dist = [
                    _imp_features[i] == _opp_imp_features[i] for i in range(len(_imp_features))
                ]
                sim = match_dist.count(True)/len(match_dist)
                sim_list.append(sim)

            max_sim = max(sim_list)
            max_sim_pos_list = [i for i, sim in enumerate(sim_list) if sim == max_sim]
            random.shuffle(max_sim_pos_list)
            max_sim_opp_imp = opp_imps[max_sim_pos_list[-1]]

            swapped_findings += imp_fd_map[max_sim_opp_imp]
            
            #random.shuffle(swapped_findings)
            #return swapped_findings[-1]

        return swapped_findings

    def _get_swap_sentence(self, fd: str, fd_sent_label_df):
        
        cls_list = fd_sent_label_df.keys()

        fd_sents = sent_tokenize(fd)
        

        final_fd = []
        for i in range(len(sent_tokenize(fd))):

            fd_sent = fd_sents[i]

            fd_sent_features = fd_sent_label_df[
                fd_sent_label_df['Reports'] == fd_sent
            ].values[0]

            pos_cls_idxs = np.where(fd_sent_features == float(1))[0]
            pos_clss = [cls_list[idx] for idx in pos_cls_idxs]
            #pos_clss = list(map(lambda x: cls_list[x], pos_cls_idxs))

            neg_cls_idxs = np.where(fd_sent_features == float(0))[0]
            neg_clss = [cls_list[idx] for idx in neg_cls_idxs]
            #neg_clss = list(map(lambda x: cls_list[x], neg_cls_idxs))

            pos_opp_fd_sent_idxes = []
            for pos_cls in pos_clss:
                curr_pos_opp_fd_sent_idxes = fd_sent_label_df[
                    fd_sent_label_df[pos_cls] == float(0)
                ].index
                pos_opp_fd_sent_idxes += list(curr_pos_opp_fd_sent_idxes)

            neg_opp_fd_sent_idxes = []
            for neg_cls in neg_clss:
                curr_neg_opp_fd_sent_idxes = fd_sent_label_df[
                    fd_sent_label_df[neg_cls] == float(1)
                ].index
                neg_opp_fd_sent_idxes += list(curr_neg_opp_fd_sent_idxes)

            opp_fd_sent_idxes = pos_opp_fd_sent_idxes + neg_opp_fd_sent_idxes
            opp_fd_sent_idxes = list(set(opp_fd_sent_idxes))
            opp_fd_sent_idxes.sort(reverse=False)

            opp_fd_sents = list(fd_sent_label_df.iloc[opp_fd_sent_idxes, 0])



    def _get_add_sentence(self, fd: str, fd_sent_label_df):
        pass

################################################################
################################################################

    def generate_swap_error(self):
        pass