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
from constants import *

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
        self, impression_label_path, finding_label_path, finding_sent_label_path, 
        class_group_info=None
        ):

        imp_label_df = pd.read_csv(impression_label_path)
        fd_label_df = pd.read_csv(finding_label_path)
        fd_sent_label_df = pd.read_csv(finding_sent_label_path)

        if class_group_info is None:
            pass    

        return imp_label_df, fd_label_df, fd_sent_label_df

    ########################################################################
    ########################################################################
    #############    FACTUAL ERROR
    ########################################################################
    ########################################################################

    def get_finding_label_df(
        self, chexbert_model, chexbert_tokenizer, batch_size, device,
        findings
    ):
        y_pred = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            findings
        )
        features_list = self._sample_wise(y_pred)
        features_list = self._map_chexbert_to_chexpert_label(features_list)

        assert len(findings) == len(features_list)

        # Apply the tree structure of the CheXpert labeler
        for features in features_list:
           self._post_process(features)

        findings_df = pd.DataFrame(findings, columns = [REPORTS])
        features_df = pd.DataFrame(features_list, columns = CONDITIONS)
        finding_label_df = pd.concat([findings_df, features_df], axis=1, join='inner')

        return finding_label_df

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
        self, embedding_matrix, text_index_map,
        finding: str, impression: str, 
        chexbert_model, chexbert_tokenizer, batch_size, device,
        swap_prob
        ):

        generated_errors = {
            'under_reading': [],
            'satisfaction_of_search': [],
        }

        p_error_cand_list = []
        for _ in range(2*batch_size - 1):
            p_error_cand = self._get_p_error_candidate(
                embedding_matrix, text_index_map,
                finding, impression,
                swap_prob
                )
            if len(p_error_cand) > 5:
                p_error_cand_list.append(p_error_cand)

        inputs = []
        inputs.append(finding)
        for p_error_cand in p_error_cand_list:
            inputs.append(p_error_cand)

        y_pred = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            inputs
        )
        features_list = self._sample_wise(y_pred)
        features_list = self._map_chexbert_to_chexpert_label(features_list)

        # Apply the tree structure of the CheXpert labeler
        for features in features_list:
           self._post_process(features)
        # Exclude "No Finding"
        _features_list = [features[:-1] for features in features_list]

        _fd_features = _features_list[0]
        _p_error_features_list = _features_list[1:]

        # If there is no positive class, cannot generate perceptual error
        fd_pos_idxs = self._get_idx_of_positive_label([_fd_features])[0]
        if len(fd_pos_idxs) == 0:
            return generated_errors

        under_readings = []
        satisfaction_of_searchs = []
        for i in range(len(_p_error_features_list)):
            if self._is_underreading(_p_error_features_list[i]):
                under_readings.append(p_error_cand_list[i])
                #under_readings += [(p_error_cand_list[i], _p_error_features_list[i])]

            if self._is_satisfaction_of_search(_fd_features, _p_error_features_list[i]):
                satisfaction_of_searchs.append(p_error_cand_list[i])
                #satisfaction_of_searchs += [(p_error_cand_list[i], _p_error_features_list[i])]


        if len(under_readings) > 0:
            random.shuffle(list(set(under_readings)))
            generated_errors['under_reading'] += [under_readings[-1]]
        if len(satisfaction_of_searchs) > 0:
            random.shuffle(list(set(satisfaction_of_searchs)))
            generated_errors['satisfaction_of_search'] += [satisfaction_of_searchs[-1]]

        return generated_errors

    def _post_process(self, features):

        # if child's value is uncertain, let parent's value be uncertain
        for idx in range(len(features)):
            parents = self._get_all_parents(CONDITIONS[idx], CHEXBERT_RELATION_PAIRS)
            if len(parents) > 0:
                if float(features[idx]) == float(-1):
                    for parent in parents:
                        features[CONDITIONS_DICT[parent]] = float(-1)                

        # if child's value is positive, let parent's value be positive
        for idx in range(len(features)):
            parents = self._get_all_parents(CONDITIONS[idx], CHEXBERT_RELATION_PAIRS)
            if len(parents) > 0:
                if float(features[idx]) == float(1):
                    for parent in parents:
                        features[CONDITIONS_DICT[parent]] = float(1)


    def _get_all_parents(self, child, relation_list):
        
        def get_parents(child, relation_list):
            parents = []
            for rel in relation_list:
                p, c = rel[0], rel[1]
                if c == child:
                    parents.append(p)
            
            return parents

        all_parents = []
        curr_parents = get_parents(child, relation_list)
        all_parents += curr_parents

        while len(curr_parents) > 0:
            next_parents = []
            for parent in curr_parents:
                next_parents += get_parents(parent, relation_list)
            all_parents += next_parents

            curr_parents = next_parents

        return all_parents

    def _get_idx_of_positive_label(self, features_list):
        pos_idxs_list = []
        for features in features_list:
            pos_idxs = [idx for idx in range(len(features)) if features[idx] == float(1)]
            pos_idxs_list.append(pos_idxs)
        
        return pos_idxs_list

    def _is_underreading(self, p_error_features):
        for idx in range(len(p_error_features)):
            if p_error_features[idx] == float(1):
                    return False

        return True

    def _is_satisfaction_of_search(self, fd_features, p_error_features):
        fd_pos_idxs = self._get_idx_of_positive_label([fd_features])[0]
        
        # Exclude the case of "faulty reasoning"
        for idx in range(len(p_error_features)):
            if idx not in fd_pos_idxs:
                if p_error_features[idx] == float(1):
                    return False

        _num_of_pos = 0
        for pos_idx in fd_pos_idxs:
            if p_error_features[pos_idx] == float(1):
                _num_of_pos += 1

        if 0 < _num_of_pos and _num_of_pos < len(fd_pos_idxs):
            return True
        else:
            return False

    def _get_p_error_candidate(
        self, embedding_matrix, text_index_map,
        finding: str, impression: str, swap_prob
        ):

        f_sents = sent_tokenize(finding)
        sim_dist = []
        for f_sent in f_sents:
            sim_dist.append(
                self.cos_sim(
                    embedding_matrix[text_index_map[impression]], 
                    embedding_matrix[text_index_map[f_sent]])
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

        return final_finding

    def cos_sim(self, vector1, vector2):
        return np.dot(vector1, vector2) / (norm(vector1)*norm(vector2))

    def _chexbert_forward(
            self, chexbert_model, chexbert_tokenizer, 
            cm_batch_size, device,
            p_error_cand_list
        ):

        with torch.no_grad():
            y_pred = [[] for _ in range(len(CONDITIONS))]
            # y_pred = (batch_size, 14)

            for idx in range(0, len(p_error_cand_list), cm_batch_size):
                mini_batch = chexbert_tokenizer(
                    p_error_cand_list[idx: idx + cm_batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                # input_ids = torch.tensor(mini_batch['input_ids'], dtype=torch.long)
                # attention_mask = torch.tensor(mini_batch['attention_mask'], dtype=torch.long)
                
                input_ids = mini_batch['input_ids'].to(device)
                attention_mask = mini_batch['attention_mask'].to(device)

                output = chexbert_model(
                    source_padded=input_ids,
                    attention_mask=attention_mask,
                )
                for i in range(len(output)):
                    curr_y_pred = output[i].argmax(dim=1)
                    y_pred[i].append(curr_y_pred)

            for i in range(len(y_pred)):
                y_pred[i] = torch.cat(y_pred[i], dim=0)
        
        y_pred = [t.tolist() for t in y_pred]
        return y_pred

    def _sample_wise(self, y_pred):

        num_of_sample = len(y_pred[0])
        sample_wise_y_pred = [[] for _ in range(num_of_sample)]
        for i in range(num_of_sample):
            for j in range(len(y_pred)):
                sample_wise_y_pred[i].append(y_pred[j][i])
            
        return sample_wise_y_pred

    def _map_chexbert_to_chexpert_label(self, sample_wise_y_pred):

        num_of_conditions = len(sample_wise_y_pred[0])
        for i in range(len(sample_wise_y_pred)):
            for j in range(num_of_conditions):
                if sample_wise_y_pred[i][j] == 0:
                    sample_wise_y_pred[i][j] = np.nan
                elif sample_wise_y_pred[i][j] == 3:
                    sample_wise_y_pred[i][j] = -1
                elif sample_wise_y_pred[i][j] == 2:
                    sample_wise_y_pred[i][j] = 0

        for i in range(len(sample_wise_y_pred)):
            sample_wise_y_pred[i] = list(map(float, sample_wise_y_pred[i]))

        return sample_wise_y_pred

    def get_embedding_matrix(self, embedding_cache_path):
        import pickle
        
        with open(embedding_cache_path, "rb") as f:
            cache_data = pickle.load(f)
            text = cache_data['text']
            embeddings = cache_data['embeddings']

        assert len(text) == len(embeddings)

        embedding_matrix = []
        text_index_map = {}
        for idx in range(len(text)):
            embedding_matrix.append(embeddings[idx])
            text_index_map[text[idx]] = idx

        return embedding_matrix, text_index_map

#    def get_swaped_sentence(self, sent, doc, docs):
#        # 동일 Finding 내의 문장 X
#        # n개의 candidate 뽑고 개별적인 cosine_similarity 계산 후 가장 낮은 랭킹의 문장 반환
#        while True:
#            random_idx = random.randrange(len(docs))
#            random_sent = docs[random_idx]
#            if random_sent != sent:
#
#                return random_sent

    # def generate_interpretive_error(
    #     self, finding: str, impression: str,
    #     imp_fd_map: dict, 
    #     imp_label_df, fd_sent_label_df,
    #     sf_prob: float , ss_prob: float, as_prob: float
    # ):
    def _generate_interpretive_error(
        self, finding: str, finding_label_df, fd_sent_label_df,
        chexbert_model, chexbert_tokenizer, batch_size, device,
        sf_prob: float , ss_prob: float, as_prob: float
    ):
        # 서로 다른 finding이 같은 impression을 갖는 경우 존재
        # chexpert_labeler 결과, impression만 존재하며 finding은 없음 -> 추후 mapping 작업도 필요

        generated_errors = {
            'faulty_reasoning_1': [],
            'faulty_reasoning_2': [],
            'complacency': [],
            'original': []
        }

        faulty_reasoning_1_list = self._get_faulty_reasoning_1(
            finding, finding_label_df
        )

        if len(faulty_reasoning_1_list) > 0:
            #if np.random.choice([True, False], 1, p=[sf_prob, 1-sf_prob]).item():
            # shuffling numerical_errors
            random.shuffle(faulty_reasoning_1_list)
            generated_errors['faulty_reasoning_1'] += [faulty_reasoning_1_list[-1]]

            # fd_features = finding_label_df[
            #     finding_label_df[REPORTS] == finding
            # ].values[0][1:]
            # self._post_process(fd_features[:-1])
            # generated_errors['original'] = [(finding, fd_features)]

        # self._get_faulty_reasoning_1의 경우 판독문 단위의 수정이므로 그 자체로 final errors
        # self._get_faulty_reasoning_2 및 self._get_complacency의 경우 문장 단위의 수정이므로,
        # candidate 생성 후 CheXbert 적용하여 검증 필요
        # faulty_reasoning_2_list = self._get_faulty_reasoning_2(
        #     finding, fd_sent_label_df,
        #     chexbert_model, chexbert_tokenizer, batch_size, device
        # )

        # if len(faulty_reasoning_2_list) > 0:
        #     #if np.random.choice([True, False], 1, p=[sf_prob, 1-sf_prob]).item():
        #     # shuffling numerical_errors
        #     random.shuffle(faulty_reasoning_2_list)
        #     generated_errors['faulty_reasoning_2'] += [faulty_reasoning_2_list[-1]]

        #     fd_features = finding_label_df[
        #         finding_label_df[REPORTS] == finding
        #     ].values[0][1:]
        #     self._post_process(fd_features)
        #     generated_errors['original'] = [(finding, fd_features[:-1])]

        # added_sentences = self._get_complacency(
        #     finding,
        #     fd_sent_label_df
        #     )

        return generated_errors

    def _get_faulty_reasoning_1(
        self, finding: str, finding_label_df
        ):

        cond_list = finding_label_df.keys()

        fd_features = finding_label_df[
            finding_label_df[REPORTS] == finding
        ].values[0]

        # if there is not finding, cannot generate faulty_reasoning_1 error
        # else exclude "No Finding" class
        if fd_features[-1] != float(1):
            return []
        _fd_features = fd_features[:-1]

        # 1. pos + neg -> 합집합O, 교집합X
        # 2. 50%/50%: ~fuzzy(=random) OR fuzzy func. 기준 유사도 높은 candidates(but label 상이) 추출

        pos_cond_idxs = np.where(_fd_features == float(1))[0]
        pos_conds = [cond_list[idx] for idx in pos_cond_idxs]
        #pos_conds = list(map(lambda x: cond_list[x], pos_cond_idxs))

        neg_cond_idxs = np.where(_fd_features == float(0))[0]
        neg_conds = [cond_list[idx] for idx in neg_cond_idxs]
        #neg_conds = list(map(lambda x: cond_list[x], neg_cond_idxs))

        pos_opp_fd_idxes = []
        for pos_cond in pos_conds:
            curr_pos_opp_fd_idxes = finding_label_df[
                finding_label_df[pos_cond] == float(0)
            ].index
            pos_opp_fd_idxes += list(curr_pos_opp_fd_idxes)

        neg_opp_fd_idxes = []
        for neg_cond in neg_conds:
            curr_neg_opp_fd_idxes = finding_label_df[
                finding_label_df[neg_cond] == float(1)
            ].index
            neg_opp_fd_idxes += list(curr_neg_opp_fd_idxes)

        opp_fd_idxes = []
        set_pos_opp_fd_idxes = set(pos_opp_fd_idxes)
        set_neg_opp_fd_idxes = set(neg_opp_fd_idxes)
        for idx in set_pos_opp_fd_idxes:
            if idx in set_neg_opp_fd_idxes:
                opp_fd_idxes.append(idx)
        opp_fd_idxes = list(set(opp_fd_idxes))
        opp_fd_idxes.sort(reverse=False)
        #print(f"{len(opp_fd_idxes)} of idxes searched")
        # error_df = finding_label_df.iloc[opp_fd_idxes]
        # error_fd_list = error_df.loc[:, REPORTS].values.tolist()
        # error_features_list = error_df.loc[:, CONDITIONS].values.tolist()
        # # Exclude "No Finding
        # _error_features_list = [error_features[:-1] for error_features in error_features_list]

        if len(opp_fd_idxes) < 1:
            return []

        #select_method = np.random.choice(['random', 'similarity'], 1, p=[0.5, 0.5]).item()
        #select_method = np.random.choice(['random', 'similarity'], 1, p=[0.0, 1.0]).item()
        select_method = np.random.choice(['random', 'similarity'], 1, p=[1.0, 0.0]).item()

        if select_method == 'random':
            random_idx = random.randrange(len(opp_fd_idxes))
            opp_fd_idx = opp_fd_idxes[random_idx]
            error_fd = finding_label_df.iloc[opp_fd_idx][REPORTS]
            error_features = finding_label_df.iloc[opp_fd_idx][CONDITIONS]
            #print(f"{len(pos_opp_fd_idxes)}||{len(neg_opp_fd_idxes)}||{len(opp_fd_idxes)} indexes searched")
            return [error_fd]
            #return [(error_fd_list[-1], _error_features_list[-1])]

        # elif select_method == 'similarity':
        #     opp_imp_features_list = imp_label_df.iloc[opp_imp_idxes, :].values
        #     _opp_imp_features_list = [
        #         list(map(str, opp_imp_features[1:])) for opp_imp_features in opp_imp_features_list
        #     ]

        #     _imp_features = list(map(str, imp_features[1:]))

        #     sim_list = []
        #     for _opp_imp_features in _opp_imp_features_list:
        #         match_dist = [
        #             _imp_features[i] == _opp_imp_features[i] for i in range(len(_imp_features))
        #         ]
        #         sim = match_dist.count(True)/len(match_dist)
        #         sim_list.append(sim)

        #     max_sim = max(sim_list)
        #     max_sim_pos_list = [i for i, sim in enumerate(sim_list) if sim == max_sim]
        #     random.shuffle(max_sim_pos_list)
        #     max_sim_opp_imp = opp_imps[max_sim_pos_list[-1]]

        #     swapped_findings += imp_fd_map[max_sim_opp_imp]
            
        #     #random.shuffle(swapped_findings)
        #     #return swapped_findings[-1]

        return error_fd

    def _get_faulty_reasoning_2(
        self, finding: str, fd_sent_label_df,
        chexbert_model, chexbert_tokenizer, batch_size, device        
        ):
        faulty_reasoning_2_list = []

        cond_list = fd_sent_label_df.keys()

        fd_sents = sent_tokenize(finding)
        error_fd_sent_dict = {}
        for sent_id in range(len(fd_sents)):

            fd_sent_features = fd_sent_label_df[
                fd_sent_label_df[REPORTS] == fd_sents[sent_id]
            ].values[0]

            pos_cond_idxs = np.where(fd_sent_features == float(1))[0]
            pos_conds = [cond_list[idx] for idx in pos_cond_idxs]

            neg_cond_idxs = np.where(fd_sent_features == float(0))[0]
            neg_conds = [cond_list[idx] for idx in neg_cond_idxs]

            pos_opp_fd_sent_idxes = []
            for pos_cond in pos_conds:
                curr_pos_opp_fd_sent_idxes = fd_sent_label_df[
                    fd_sent_label_df[pos_cond] == float(0)
                ].index
                pos_opp_fd_sent_idxes += list(curr_pos_opp_fd_sent_idxes)

            neg_opp_fd_sent_idxes = []
            for neg_cond in neg_conds:
                curr_neg_opp_fd_sent_idxes = fd_sent_label_df[
                    fd_sent_label_df[neg_cond] == float(1)
                ].index
                neg_opp_fd_sent_idxes += list(curr_neg_opp_fd_sent_idxes)

            opp_fd_sent_idxes = pos_opp_fd_sent_idxes + neg_opp_fd_sent_idxes
            if len(opp_fd_sent_idxes) == 0:
                continue           

            opp_fd_sent_idxes = list(set(opp_fd_sent_idxes))
            random.shuffle(opp_fd_sent_idxes)
            _opp_fd_sent_idxes = opp_fd_sent_idxes[:10]
            _opp_fd_sent_idxes.sort(reverse=False)

            error_df = fd_sent_label_df.iloc[_opp_fd_sent_idxes]
            error_fd_sent_list = error_df.loc[:, REPORTS].values.tolist()
            # error_features_list = error_df.loc[:, CONDITIONS].values.tolist()
            error_fd_sent_dict[sent_id] = error_fd_sent_list

        error_cand_list = []
        for _ in range(2*batch_size - 1):
            error_cand = []
            swap_count = 0
            for sent_id in range(len(fd_sents)):
                is_swap = np.random.choice([True, False], 1, p=[0.5, 0.5]).item()
                if sent_id in error_fd_sent_dict and is_swap:
                    error_fd_sent_list = error_fd_sent_dict[sent_id]
                    random_idx = random.randrange(len(error_fd_sent_list))
                    error_cand.append(error_fd_sent_list[random_idx])
                    swap_count += 1
                else:
                    error_cand.append(fd_sents[sent_id])

            #   3   4   5   6   7   8   9  10  11  12  13
            # 0.9 1.2 1.5 1.8 2.1 2.4 2.7 3.0 3.3 3.6 3.9  (x 0.3)
            #   1   1   2   2   2   2   3   3   3   4   4  (round)
            if round(len(error_cand)*0.3) >= swap_count:
                error_cand = " ".join(error_cand)
                error_cand_list.append(error_cand)

        inputs = []
        inputs.append(finding)
        for error_cand in error_cand_list:
            inputs.append(error_cand)

        y_pred = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            inputs
        )
        features_list = self._sample_wise(y_pred)
        features_list = self._map_chexbert_to_chexpert_label(features_list)

        # Apply the tree structure of the CheXpert labeler
        for features in features_list:
           self._post_process(features)
        # Exclude "No Finding"
        _features_list = [features[:-1] for features in features_list]

        _fd_features = _features_list[0]
        _error_features_list = _features_list[1:]

        # If there is no positive class, cannot generate perceptual error
        #fd_pos_idxs = self._get_idx_of_positive_label([_fd_features])[0]

        faulty_reasoning_2_list = []
        for i in range(len(_error_features_list)):
            if self._is_faulty_reasoning(_fd_features, _error_features_list[i]):
                #faulty_reasoning_2_list.append(error_cand_list[i])
                faulty_reasoning_2_list += [(error_cand_list[i], _error_features_list[i])]

        if len(faulty_reasoning_2_list) > 0:
            return faulty_reasoning_2_list
        else:
            return []

    def _is_faulty_reasoning(self, fd_features, error_features):
        # pos가 neg
        # neg가 pos
        # 위의 2조건이 &&로 묶여야 함.

        assert len(fd_features) == len(error_features)

        pos_neg = False
        neg_pos = False
        for f_f, e_f in zip(fd_features, error_features):
            if f_f == float(1) and e_f == float(0):
                pos_neg = True
            if f_f == float(0) and e_f == float(1):
                neg_pos = True

        if pos_neg == True and neg_pos == True:
            return True
        else:
            return False

    def _get_complacency(self, finding: str, fd_sent_label_df):
        # to generate false-positive error, there must be no finding
        # pos condition = 0인데, 일부 문장들을 추가하여 pos condition > 0 으로 만들어야 함
        # 1) finding_label_df를 argument로 받아서 "No Finding"=1인 경우를 활용할지, 나머지 모든 cond을 체크할지
        # 1-1) finding_label_df를 통해 얻은 fd_feature를 그대로 사용하고, 이후 CheXbert forward 할 때 finding은 넣지 말지
        # 1-2) 아니라면 CheXbert에 finding과 candidate error findings를 한 번에 forward할지 (이전 코드들과 일관성 높음)
        # 2) 몇개의 fd_sent를 추가할지? (min, max)
        # 3) 추가할 fd_sent들은 어떤 cond에 대한 pos인 그룹군을 사용할지
        # 3-1) 만약 n(>1)개의 fd_sent를 추가할 때, 모든 fd_sent를 동일한 cond에 대한 pos인 것으로 할지, 서로 다른 cond에 대한 pos로 구성할지
        pass

################################################################
################################################################

    def generate_swap_error(self):
        pass