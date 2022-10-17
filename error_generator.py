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
           self._convert_tree_structure(features)

        findings_df = pd.DataFrame(findings, columns = [REPORTS])
        fd_features_df = pd.DataFrame(features_list, columns = CONDITIONS)
        finding_label_df = pd.concat([findings_df, fd_features_df], axis=1, join='inner')

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
        self, embedding_matrix, text_index_map, finding_label_df,
        finding: str, impression: str,
        chexbert_model, chexbert_tokenizer, batch_size, device,
        swap_prob
        ):

        generated_errors = {
            'under_reading': [],
            'satisfaction_of_search': [],
        }

        fd_features = finding_label_df[
            finding_label_df[REPORTS] == finding
        ].values[0][1:]
        # if there is no finding, cannot generate perceptual error
        if fd_features[-1] == float(1):
            return []
        # Exclude "No Finding" label
        _fd_features = fd_features[:-1]


        error_cand_list = []
        for _ in range(3*batch_size - 1):
            error_cand = self._get_p_error_candidate(
                embedding_matrix, text_index_map,
                finding, impression,
                swap_prob
                )
            if len(error_cand) > 5:
                error_cand_list.append(error_cand)

        y_pred = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            error_cand_list
        )
        error_features_list = self._sample_wise(y_pred)
        error_features_list = self._map_chexbert_to_chexpert_label(error_features_list)

        # Apply the tree structure of the CheXpert labeler
        for error_features in error_features_list:
           self._convert_tree_structure(error_features)
        # Exclude "No Finding"
        _error_features_list = [error_features[:-1] for error_features in error_features_list]

        # If there is no positive class, cannot generate perceptual error
        fd_pos_idxs = self._get_idx_of_positive_label([_fd_features])[0]
        if len(fd_pos_idxs) == 0:
            return generated_errors

        under_readings = []
        satisfaction_of_searchs = []
        for i in range(len(_error_features_list)):
            if self._is_underreading(_error_features_list[i]):
                under_readings.append(error_cand_list[i])
                #under_readings += [(p_error_cand_list[i], _p_error_features_list[i])]

            if self._is_satisfaction_of_search(_fd_features, _error_features_list[i]):
                satisfaction_of_searchs.append(error_cand_list[i])
                #satisfaction_of_searchs += [(error_cand_list[i], _error_features_list[i])]


        if len(under_readings) > 0:
            random.shuffle(list(set(under_readings)))
            generated_errors['under_reading'] += [under_readings[-1]]
        if len(satisfaction_of_searchs) > 0:
            random.shuffle(list(set(satisfaction_of_searchs)))
            generated_errors['satisfaction_of_search'] += [satisfaction_of_searchs[-1]]

        return generated_errors

    def _convert_tree_structure(self, features):

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

    def generate_interpretive_error(
        self, finding: str, finding_label_df, fd_sent_label_df,
        chexbert_model, chexbert_tokenizer, batch_size, device,
        sf_prob: float , ss_prob: float, as_prob: float
    ):

        generated_errors = {
            'faulty_reasoning_1': [],
            'faulty_reasoning_2': [],
            'random_swap': [],
            'complacency': [],
            'original': []
        }

        fd_features = finding_label_df[
            finding_label_df[REPORTS] == finding
        ].values[0][1:]


        faulty_reasoning_1_list = self._get_faulty_reasoning_1(
            finding, fd_features, finding_label_df
        )

        if len(faulty_reasoning_1_list) > 0:
            #if np.random.choice([True, False], 1, p=[sf_prob, 1-sf_prob]).item():
            # shuffling numerical_errors
            #random.shuffle(faulty_reasoning_1_list)
            generated_errors['faulty_reasoning_1'] += [faulty_reasoning_1_list[-1]]


        faulty_reasoning_2_list = self._get_faulty_reasoning_2(
            finding, fd_features, 
            finding_label_df, fd_sent_label_df,
            chexbert_model, chexbert_tokenizer, batch_size, device
        )

        if len(faulty_reasoning_2_list) > 0:
            #if np.random.choice([True, False], 1, p=[sf_prob, 1-sf_prob]).item():
            # shuffling numerical_errors
            random.shuffle(faulty_reasoning_2_list)
            generated_errors['faulty_reasoning_2'] += [faulty_reasoning_2_list[-1]]


        random_swap_list = self._get_random_swap(
            finding, fd_features, finding_label_df
        )

        if len(random_swap_list) > 0:
            generated_errors['random_swap'] += [random_swap_list[-1]]


        complacency_list = self._get_complacency(
            finding, fd_features,
            finding_label_df, fd_sent_label_df,
            chexbert_model, chexbert_tokenizer, batch_size, device
            )

        if len(complacency_list) > 0:
            generated_errors['complacency'] += [complacency_list[-1]]

        
        generated_errors['original'] = [(finding, fd_features[:-1])]

        return generated_errors

    def _get_faulty_reasoning_1(
        self, finding: str, fd_features, finding_label_df
        ):

        # if there is no finding, cannot generate faulty_reasoning_1 error
        if fd_features[-1] == float(1):
            return []
        # Exclude "No Finding" class
        _fd_features = fd_features[:-1]

        cond_list = finding_label_df.keys()[1:]

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
        if len(opp_fd_idxes) == 0:
            return []

        opp_fd_idxes = list(set(opp_fd_idxes))
        random.shuffle(opp_fd_idxes)
        _opp_fd_idxes = opp_fd_idxes[:20]
        _opp_fd_idxes.sort(reverse=False)

        tmp_df = finding_label_df.iloc[_opp_fd_idxes]
        swap_findings = tmp_df.loc[:, REPORTS].values.tolist()
        swap_features = tmp_df.loc[:, CONDITIONS].values.tolist()
        _swap_features = [sf[:-1] for sf in swap_features]
        random_idx = random.randrange(len(swap_findings))
        
        #return swap_findings
        return [(swap_findings[random_idx], _swap_features[random_idx])]

    def _get_faulty_reasoning_2(
        self, finding: str, fd_features, 
        finding_label_df, fd_sent_label_df,
        chexbert_model, chexbert_tokenizer, batch_size, device        
        ):


        # if there is no finding, cannot generate faulty_reasoning_2 error
        if fd_features[-1] == float(1):
            return []

        cond_list = finding_label_df.keys()[1:]
        _cond_list = fd_sent_label_df.keys()[1:]
        assert cond_list.all() == _cond_list.all()

        # Exclude "No Finding" label
        _fd_features = fd_features[:-1]

        fd_pos_cond_idxs = np.where(_fd_features == float(1))[0]
        fd_pos_conds = [cond_list[idx] for idx in fd_pos_cond_idxs]

        fd_neg_cond_idxs = np.where(_fd_features == float(0))[0]
        fd_neg_conds = [cond_list[idx] for idx in fd_neg_cond_idxs]

        swap_fd_sent_dict = {}
        fd_sents = sent_tokenize(finding)
        for sent_id in range(len(fd_sents)):

            fd_sent_features = fd_sent_label_df[
                fd_sent_label_df[REPORTS] == fd_sents[sent_id]
            ].values[0][1:]
            # Exclude "No Finding"
            _fd_sent_features = fd_sent_features[:-1]

            fd_sent_pos_cond_idxs = np.where(_fd_sent_features == float(1))[0]
            fd_sent_pos_conds = [cond_list[idx] for idx in fd_sent_pos_cond_idxs]
            pos_conds = [fd_sent_pos_cond for fd_sent_pos_cond in fd_sent_pos_conds if fd_sent_pos_cond in fd_pos_conds]

            fd_sent_neg_cond_idxs = np.where(_fd_sent_features == float(0))[0]
            fd_sent_neg_conds = [cond_list[idx] for idx in fd_sent_neg_cond_idxs]
            neg_conds = [fd_sent_neg_cond for fd_sent_neg_cond in fd_sent_neg_conds if fd_sent_neg_cond in fd_neg_conds]

            if len(pos_conds) == 0 and len(neg_conds) == 0:
                continue


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
            opp_fd_sent_idxes = list(set(opp_fd_sent_idxes))
            if len(opp_fd_sent_idxes) > 0:
                random.shuffle(opp_fd_sent_idxes)
                _opp_fd_sent_idxes = opp_fd_sent_idxes[:10]
                _opp_fd_sent_idxes.sort(reverse=False)

                tmp_df = fd_sent_label_df.iloc[_opp_fd_sent_idxes]
                swap_fd_sent_list = tmp_df.loc[:, REPORTS].values.tolist()
                # swap_features_list = tmp_df.loc[:, CONDITIONS].values.tolist()
                swap_fd_sent_dict[sent_id] = swap_fd_sent_list

        error_cand_list = []
        for _ in range(3*batch_size):
            error_cand = []       
            swap_count = 0
            for sent_id in range(len(fd_sents)):
                is_swap = np.random.choice([True, False], 1, p=[0.5, 0.5]).item()
                if sent_id in swap_fd_sent_dict and is_swap:
                    swap_fd_sent_list = swap_fd_sent_dict[sent_id]
                    random_idx = random.randrange(len(swap_fd_sent_list))
                    error_cand.append(swap_fd_sent_list[random_idx])
                    swap_count += 1
                else:
                    error_cand.append(fd_sents[sent_id])

            #   3   4   5   6   7   8   9  10  11  12  13
            # 0.9 1.2 1.5 1.8 2.1 2.4 2.7 3.0 3.3 3.6 3.9  (x 0.3)
            #   1   1   2   2   2   2   3   3   3   4   4  (round)
            if round(len(error_cand)*0.3) >= swap_count:
                error_cand = " ".join(error_cand)
                error_cand_list.append(error_cand)

        y_pred = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            error_cand_list
        )
        error_features_list = self._sample_wise(y_pred)
        error_features_list = self._map_chexbert_to_chexpert_label(error_features_list)
        for error_features in error_features_list:
           self._convert_tree_structure(error_features)

        # Exclude "No Finding"
        _error_features_list = [error_features[:-1] for error_features in error_features_list]

        faulty_reasoning_2_list = []
        for i in range(len(_error_features_list)):
            if self._is_faulty_reasoning(_fd_features, _error_features_list[i]):
                #faulty_reasoning_2_list.append(error_cand_list[i])
                faulty_reasoning_2_list += [(error_cand_list[i], _error_features_list[i])]

        if len(faulty_reasoning_2_list) > 0:
            return faulty_reasoning_2_list
        else:
            return []

    def _is_faulty_reasoning(self, _fd_features, _error_features):
        # pos가 neg
        # neg가 pos
        # 위의 2조건이 &로 묶여야 함.

        assert len(_fd_features) == len(_error_features)

        pos_neg = False
        neg_pos = False
        for f_f, e_f in zip(_fd_features, _error_features):
            if f_f == float(1) and e_f == float(0):
                pos_neg = True
            if f_f == float(0) and e_f == float(1):
                neg_pos = True

        if pos_neg == True and neg_pos == True:
            return True
        else:
            return False

    def _get_complacency(
        self, finding: str, fd_features, 
        finding_label_df, fd_sent_label_df,
        chexbert_model, chexbert_tokenizer, batch_size, device
        ):
        
        cond_list = finding_label_df.keys()[1:]
        _cond_list = fd_sent_label_df.keys()[1:]
        assert cond_list.all() == _cond_list.all()

        # Exclude "No Finding" label
        _fd_features = fd_features[:-1]

        neg_cond_idxs = np.where(_fd_features == float(0))[0]
        neg_conds = [cond_list[idx] for idx in neg_cond_idxs]

        # nan_cond_idxs = np.where(_fd_features == np.nan)[0]
        # is_nan = np.isnan(_fd_features.astype(float))
        # nan_cond_idxs = np.where(is_nan == True)[0]
        #__fd_features = np.array(list(map(lambda x: None if np.isnan(x) else x), _fd_features))
        __fd_features = np.array(list(map(lambda x: None if np.isnan(x) else x, _fd_features)))
        nan_cond_idxs = np.where(__fd_features == None)[0]
        nan_conds = [cond_list[idx] for idx in nan_cond_idxs]

        # if there is no neg or nan label, cannot generate complacency error
        if len(neg_conds) == 0 and len(nan_conds) == 0:
            return []
        
        conds = neg_conds + nan_conds
        random_idx = random.randrange(len(conds))
        target_cond = conds[random_idx]
        if target_cond in neg_conds:
            target_cond_label = float(0)
        elif target_cond in nan_conds:
            target_cond_label = np.nan
        else:
            raise NotImplementedError

        target_opp_dict = {}
        fd_sents = sent_tokenize(finding)
        for sent_id in range(len(fd_sents)):

            fd_sent_features = fd_sent_label_df[
                fd_sent_label_df[REPORTS] == fd_sents[sent_id]
            ].values[0][1:]
            # Exclude "No Finding" label
            _fd_sent_features = fd_sent_features[:-1]

            target_cond_idx = CONDITIONS_DICT[target_cond]
            if np.isnan(target_cond_label) and not np.isnan(_fd_sent_features[target_cond_idx]):
                continue
            if target_cond_label == float(0) and _fd_sent_features[target_cond_idx] != target_cond_label:
                continue

            target_opp_idxes = fd_sent_label_df[
                fd_sent_label_df[target_cond] == float(1)
            ].index.tolist()

            random.shuffle(target_opp_idxes)
            _target_opp_idxes = target_opp_idxes[:10]
            _target_opp_idxes.sort(reverse=False)

            tmp_df = fd_sent_label_df.iloc[target_opp_idxes]
            target_opp_fd_sent_list = tmp_df.loc[:, REPORTS].values.tolist()
            # target_opp_features_list = tmp_df.loc[:, CONDITIONS].values.tolist()
            target_opp_dict[sent_id] = target_opp_fd_sent_list

        target_opp_fd_sent_list = []
        for _, curr_fd_sent_list in target_opp_dict.items():
            target_opp_fd_sent_list += curr_fd_sent_list

        if len(target_opp_fd_sent_list) == 0:
            return []

        target_opp_fd_sent_list = list(set(target_opp_fd_sent_list))

        max_num_of_insert = 1
        error_cand_list = []
        for _ in range(3*batch_size):
            error_cand = []
            for sent_id in range(len(fd_sents)):
                if target_cond_label == float(0) and sent_id in target_opp_dict:
                    continue
                else:
                    error_cand.append(fd_sents[sent_id])

            if len(error_cand) > 0:
                for _ in range(max_num_of_insert):
                    insert_position = random.randrange(len(error_cand))
                    insert_fd_sent = target_opp_fd_sent_list[random.randrange(len(target_opp_fd_sent_list))]
                    error_cand.insert(insert_position, insert_fd_sent)

                error_cand = " ".join(error_cand)
                error_cand_list.append(error_cand)

        if len(error_cand_list) == 0:
            print(1)
            return []

        y_pred = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            error_cand_list
        )
        error_features_list = self._sample_wise(y_pred)
        error_features_list = self._map_chexbert_to_chexpert_label(error_features_list)
        for error_features in error_features_list:
            self._convert_tree_structure(error_features)

        # Exclude "No Finding"
        _error_features_list = [error_features[:-1] for error_features in error_features_list]

        complacency_list = []
        for i in range(len(_error_features_list)):
            if self._is_complacency(target_cond, _error_features_list[i]):
                #complacency_list.append(error_cand_list[i])
                complacency_list += [(error_cand_list[i], _error_features_list[i])]

        if len(complacency_list) > 0:
            return complacency_list
        else:
            return []

    def _is_complacency(self, target_cond, c_error_features):
        target_cond_idx = CONDITIONS_DICT[target_cond]
        if c_error_features[target_cond_idx] == float(1):
            return True
        else:
            return False

    # def _get_random_swap(
    #     self, finding: str, fd_features, finding_label_df
    #     ):

    #     cond_list = finding_label_df.keys()[1:]

    #     # Exclude "No Finding" class
    #     _fd_features = fd_features[:-1]
        
    #     label_list = [float(1), float(0), np.nan]

    #     curr_label_list = set(_fd_features)
    #     print(curr_label_list)
    #     source_label = np.random.choice(curr_label_list, 1,replace=False).item()
    #     source_cond_idxs = np.where(_fd_features == source_label)[0]
    #     source_conds = [cond_list[idx] for idx in source_cond_idxs]

    #     label_list.remove(source_label)
    #     target_label = np.random.choice(label_list, 1,replace=False).item()

    #     target_label = np.nan
    #     # max_iter = 2 * len(label_list)
    #     # current_iter = 0
    #     # while True:
    #     #     source_label = np.random.choice(label_list, 1,replace=False).item()
    #     #     source_cond_idxs = np.where(_fd_features == source_label)[0]
    #     #     source_conds = [cond_list[idx] for idx in source_cond_idxs]
    #     #     if len(source_conds) > 0:
    #     #         break

    #     #     current_iter += 1
    #     #     if current_iter > max_iter:
    #     #         return []

    #     # label_list.remove(source_label)
    #     # target_label = np.random.choice(label_list, 1,replace=False).item()

    #     target_fd_idxes = []
    #     for source_cond in source_conds:
    #         curr_target_fd_idxes = finding_label_df[
    #             finding_label_df[source_cond] == target_label
    #         ].index
    #         target_fd_idxes += list(curr_target_fd_idxes)

    #     if len(target_fd_idxes) == 0:
    #         return []

    #     target_fd_idxes = list(set(target_fd_idxes))
    #     random.shuffle(target_fd_idxes)
    #     _target_fd_idxes = target_fd_idxes[:20]
    #     _target_fd_idxes.sort(reverse=False)

    #     tmp_df = finding_label_df.iloc[_target_fd_idxes]
    #     swap_findings = tmp_df.loc[:, REPORTS].values.tolist()
    #     swap_features = tmp_df.loc[:, CONDITIONS].values.tolist()
    #     _swap_features = [sf[:-1] for sf in swap_features]
    #     random_idx = random.randrange(len(swap_findings))
        
    #     #return swap_findings
    #     return [(swap_findings[random_idx], _swap_features[random_idx])]

    def _get_random_swap(
        self, finding: str, fd_features, finding_label_df
        ):

        # Exclude "No Finding" class
        _fd_features = fd_features[:-1]

        random_idxes_set = set()
        while len(random_idxes_set) < 50:
            random_idx = random.randrange(finding_label_df.shape[0])
            random_idxes_set.add(random_idx)
        random_idxes = list(random_idxes_set)
        random_idxes.sort(reverse=False)

        tmp_df = finding_label_df.iloc[random_idxes]
        random_findings = tmp_df.loc[:, REPORTS].values.tolist()
        random_features_list = tmp_df.loc[:, CONDITIONS].values.tolist()
        _random_features_list = [rf[:-1] for rf in random_features_list]

        swap_findings = []
        _swap_features_list = []
        for i in range(len(random_idxes)):
            _random_features = _random_features_list[i]

            # to compare ndarray, need to convert np.nan to None
            __fd_features = np.array(list(map(lambda x: None if np.isnan(x) else x, _fd_features)))
            __random_features = np.array(list(map(lambda x: None if np.isnan(x) else x, _random_features)))

            if (__fd_features == __random_features).all() != True:
                swap_findings.append(random_findings[i])
                _swap_features_list.append(_random_features)

        if len(swap_findings) > 0:
            random_idx = random.randrange(len(swap_findings))
            return [(swap_findings[random_idx], _swap_features_list[random_idx])]
        else:
            return []


    def _get_idx_of_positive_label(self, features_list):
        pos_idxs_list = []
        for features in features_list:
            pos_idxs = [idx for idx in range(len(features)) if features[idx] == float(1)]
            pos_idxs_list.append(pos_idxs)
        
        return pos_idxs_list