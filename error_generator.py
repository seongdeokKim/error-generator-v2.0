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

        self.original_samples = []
        self.original_findings = []
        self.original_finding_sents = []
        self.original_impressions = []

        self.condition_list = CONDITIONS

    def load_original_samples(self, input_path):
        self.original_samples = [json.loads(l) for l in open(input_path)]
        for sample in self.original_samples:
            self.original_findings.append(sample['Findings'])
            self.original_impressions.append(sample['Impression'])
            self.original_finding_sents += sent_tokenize(sample['Findings'])

    def get_chexpert_label_of_finding_sents(self, finding_sent_label_path):

        data_df = pd.read_csv(finding_sent_label_path)
        data = data_df.values.tolist()

        column_names = []
        column_names += [REPORTS]
        column_names += CONDITIONS

        fd_sent_label_df = pd.DataFrame(data, columns=column_names)

        return fd_sent_label_df

    def get_chexbert_label_of_finding(
        self, chexbert_model, chexbert_tokenizer, batch_size, device,
        findings
    ):
        features_list = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            findings
        )
        # Apply the tree structure of the CheXpert labeler
        for features in features_list:
           self._convert_tree_structure(features)

        findings_df = pd.DataFrame(findings, columns=[REPORTS])
        fd_features_df = pd.DataFrame(features_list, columns=CONDITIONS)
        finding_label_df = pd.concat([findings_df, fd_features_df], axis=1, join='inner')

        return finding_label_df

    def generate_factual_miss(
        self, finding: str,
        n_prob=0.8, u_prob=0.8, l_prob=0.1
        ):

        generated_errors = {
            'numerical_error': [],
            'unit_error': [],
            'laterality_error': [],
        }

        if np.random.choice([True, False], 1, p=[n_prob, 1-n_prob]).item():
            numerical_errors = self._get_numerical_error(finding)
            if len(numerical_errors) > 0:
                random_idx = random.randrange(len(numerical_errors))
                generated_errors['numerical_error'] += [numerical_errors[random_idx]]

        if np.random.choice([True, False], 1, p=[u_prob, 1-u_prob]).item():
            unit_errors = self._get_unit_error(finding)
            if len(unit_errors) > 0:
                random_idx = random.randrange(len(unit_errors))
                generated_errors['unit_error'] += [unit_errors[random_idx]]

        if np.random.choice([True, False], 1, p=[l_prob, 1-l_prob]).item():
            laterality_errors = self._get_laterality_error(finding)
            if len(laterality_errors) > 0:
                random_idx = random.randrange(len(laterality_errors))
                generated_errors['laterality_error'] += [laterality_errors[random_idx]]

        return generated_errors

    def _get_numerical_error(self, finding: str):

        numerical_errors = []
        
        numeric_units = re.findall(r'(?:\d+(?:\.\d+)?(?::?\d+)?)(?:\s*[a|p|c]?\.?m\.?m?)', finding)
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

        tmp_finding = finding
        for i in range(len(numeric_units)):
            numerical_errors.append(
                tmp_finding.replace(numeric_units[i], generated[i])
            )

        return numerical_errors

    def _get_unit_error(self, finding: str):
        unit_errors = []

        numeric_units = re.findall(r'(?:\d+(?:\.\d+)?(?::?\d+)?)(?:\s*[a|p|c]?\.?m\.?m?)', finding)
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

        tmp_finding = finding
        for i in range(len(numeric_units)):
            unit_errors.append(
                tmp_finding.replace(numeric_units[i], generated[i])
            )

        return unit_errors

    def _get_laterality_error(self, finding: str):

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
            if key1 in finding.lower() or key2 in finding.lower():
                pair_dicts.append(laterality_dict[key_pair])

        if len(pair_dicts) == 0:
            return []

        laterality_errors = []
        for i in range(len(pair_dicts)):
            pair_dict = pair_dicts[i]
            regex = re.compile("(%s)" % "|".join(map(re.escape, pair_dict.keys())))
            laterality_error = regex.sub(
                lambda mo: pair_dict[mo.string[mo.start():mo.end()]], 
                finding
            )
            laterality_errors.append(laterality_error)

        return laterality_errors
        
    def generate_under_reading(
        self, embedding_matrix, text_index_map, finding_label_df,
        finding: str, impression: str,
        chexbert_model, chexbert_tokenizer, batch_size, device,
        del_prob
        ):

        generated_errors = {
            'under_reading': [],
        }

        fd_features = self._get_features(finding_label_df, finding)
        # if there is no finding, cannot generate perceptual error
        if fd_features[NO_FINDING_IDX] == POSITIVE:
            return generated_errors

        # Exclude "No Finding" label
        _fd_features = fd_features[:NO_FINDING_IDX]

        # Compute the similarity of each finding sentence with impression
        fd_sents = sent_tokenize(finding)
        sim_list = []
        for fd_sent in fd_sents:
            sim_list.append(
                self.cos_sim(
                    embedding_matrix[text_index_map[impression]], 
                    embedding_matrix[text_index_map[fd_sent]])
            )

        # get index of finding sentence which has high similarity to impression
        # select top 50% sentence from all sentences
        num_of_sents = len(sim_list)
        tmp_sim_list = list(sim_list)
        top_sim_idx_list = []
        while len(top_sim_idx_list) < max(1, round(num_of_sents * 0.5)):
            curr_highest = max(tmp_sim_list)
            tmp_sim_list.remove(curr_highest)

            top_sim_idx_list.append(sim_list.index(curr_highest))

        error_fd_cand_list = []
        for _ in range(batch_size):
            error_fd_cand = self._delete_top_k_finding_sents(
                top_sim_idx_list,
                finding,
                del_prob
            )
            if len(error_fd_cand) > 10:
                error_fd_cand_list.append(error_fd_cand)

        if len(error_fd_cand_list) == 0:
            return generated_errors


        error_fd_features_list = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            error_fd_cand_list
        )
        # Apply the tree structure of the CheXpert labeler
        for error_fd_features in error_fd_features_list:
           self._convert_tree_structure(error_fd_features)
        # Exclude "No Finding"
        _error_fd_features_list = [error_fd_features[:NO_FINDING_IDX] for error_fd_features in error_fd_features_list]

        # If there is no positive class, cannot generate perceptual error
        fd_pos_idxs = self._get_idx_of_positive_label([_fd_features])[0]
        if len(fd_pos_idxs) == 0:
            return generated_errors

        under_readings = []
        for i in range(len(_error_fd_features_list)):
            if self._is_underreading(_error_fd_features_list[i]):
                #under_readings.append(error_cand_list[i])
                under_readings += [(error_fd_cand_list[i], _error_fd_features_list[i])]

        if len(under_readings) > 0:
            random_idx = random.randrange(len(under_readings))
            generated_errors['under_reading'] += [under_readings[random_idx]]

        return generated_errors

    def generate_satisfaction_of_search(
        self, embedding_matrix, text_index_map, finding_label_df,
        finding: str, impression: str,
        chexbert_model, chexbert_tokenizer, batch_size, device,
        del_prob
        ):

        generated_errors = {
            'satisfaction_of_search': [],
        }

        fd_features = self._get_features(finding_label_df, finding)
        # if there is no finding, cannot satisfaction_of_search error
        if fd_features[NO_FINDING_IDX] == POSITIVE:
            return generated_errors

        # Exclude "No Finding" label
        _fd_features = fd_features[:NO_FINDING_IDX]

        # Compute the similarity of each finding sentence with impression
        fd_sents = sent_tokenize(finding)
        sim_list = []
        for fd_sent in fd_sents:
            sim_list.append(
                self.cos_sim(
                    embedding_matrix[text_index_map[impression]], 
                    embedding_matrix[text_index_map[fd_sent]])
            )

        # get index of finding sentence which has high similarity to impression
        # select top 50% sentence from all sentences
        num_of_sents = len(sim_list)
        tmp_sim_list = list(sim_list)
        top_sim_idx_list = []
        while len(top_sim_idx_list) < max(1, round(num_of_sents * 0.5)):
            curr_highest = max(tmp_sim_list)
            tmp_sim_list.remove(curr_highest)

            top_sim_idx_list.append(sim_list.index(curr_highest))

        error_fd_cand_list = []
        for _ in range(batch_size):
            error_fd_cand = self._delete_top_k_finding_sents(
                top_sim_idx_list,
                finding,
                del_prob
            )
            if len(error_fd_cand) > 10:
                error_fd_cand_list.append(error_fd_cand)

        if len(error_fd_cand_list) == 0:
            return generated_errors

        error_fd_features_list = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            error_fd_cand_list
        )
        # Apply the tree structure of the CheXpert labeler
        for error_fd_features in error_fd_features_list:
           self._convert_tree_structure(error_fd_features)
        # Exclude "No Finding"
        _error_fd_features_list = [error_fd_features[:NO_FINDING_IDX] for error_fd_features in error_fd_features_list]

        # If there is no positive class, cannot generate perceptual error
        fd_pos_idxs = self._get_idx_of_positive_label([_fd_features])[0]
        if len(fd_pos_idxs) == 0:
            return generated_errors

        satisfaction_of_searchs = []
        for i in range(len(_error_fd_features_list)):
            if self._is_satisfaction_of_search(_fd_features, _error_fd_features_list[i]):
                #satisfaction_of_searchs.append(error_cand_list[i])
                satisfaction_of_searchs += [(error_fd_cand_list[i], _error_fd_features_list[i])]

        if len(satisfaction_of_searchs) > 0:
            random_idx = random.randrange(len(satisfaction_of_searchs))
            generated_errors['satisfaction_of_search'] += [satisfaction_of_searchs[random_idx]]

        return generated_errors

    def _convert_tree_structure(self, features):

        # if child condition's label is uncertain, let parent condition's label be uncertain
        for idx in range(len(features)):
            parents = self._get_all_parents(CONDITIONS[idx], TREE_STRUCTURE_PAIRS)
            if len(parents) > 0:
                if float(features[idx]) == UNCERTAIN:
                    for parent in parents:
                        features[CONDITIONS_DICT[parent]] = UNCERTAIN               

        # if child condition's label is positive, let parent condition's label be positive
        for idx in range(len(features)):
            parents = self._get_all_parents(CONDITIONS[idx], TREE_STRUCTURE_PAIRS)
            if len(parents) > 0:
                if float(features[idx]) == POSITIVE:
                    for parent in parents:
                        features[CONDITIONS_DICT[parent]] = POSITIVE


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

    def _is_underreading(self, _error_fd_features):
        for idx in range(len(_error_fd_features)):
            if _error_fd_features[idx] == POSITIVE:
                return False

        return True

    def _is_satisfaction_of_search(self, _fd_features, _error_fd_features):

        fd_pos_idxs = self._get_idx_of_positive_label([_fd_features])[0]
        num_of_pos = len(fd_pos_idxs)

        error_num_of_pos = 0
        for pos_idx in fd_pos_idxs:
            if _error_fd_features[pos_idx] == POSITIVE:
                error_num_of_pos += 1

        if 0 < error_num_of_pos < num_of_pos:
            return True
        else:
            return False

    def _delete_top_k_finding_sents(
        self, top_sim_fd_sent_id_list,
        finding: str, del_prob
        ):

        fd_sents = sent_tokenize(finding)

        error_fd_sents = []
        del_count = 0
        for sent_id in range(len(fd_sents)):
            if sent_id in top_sim_fd_sent_id_list:
                is_del = np.random.choice([True, False], 1, p=[del_prob, 1-del_prob]).item()
                if is_del:
                    #error_fd_sents.append('[ELIMINATE]')
                    del_count += 1
                else:
                    error_fd_sents.append(fd_sents[sent_id])
            else:
                error_fd_sents.append(fd_sents[sent_id])

        # if len(fd_sents)*0.4 < del_count:
        #     return False
        error_finding = " ".join(error_fd_sents)
        
        return error_finding

    def cos_sim(self, vector1, vector2):
        return np.dot(vector1, vector2) / (norm(vector1)*norm(vector2))

    def _chexbert_forward(
            self, chexbert_model, chexbert_tokenizer, 
            batch_size, device,
            error_cand_list
        ):

        chexbert_model.eval()
        with torch.no_grad():
            y_pred = [[] for _ in range(len(CONDITIONS))]
            # y_pred = (batch_size, 14)
            for idx in range(0, len(error_cand_list), batch_size):
                mini_batch = chexbert_tokenizer(
                    error_cand_list[idx: idx + batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                
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
        y_pred = np.array(y_pred, dtype=np.float64)
        batch_first_y_pred = y_pred.T
        batch_first_y_pred = self._convert_chexbert_to_chexpert_label(batch_first_y_pred)

        return batch_first_y_pred

    def _convert_chexbert_to_chexpert_label(self, batch_first_y_pred):

        num_of_conditions = len(batch_first_y_pred[0])
        for batch_idx in range(len(batch_first_y_pred)):
            for cond_idx in range(num_of_conditions):
                if batch_first_y_pred[batch_idx][cond_idx] == 0:
                    batch_first_y_pred[batch_idx][cond_idx] = np.nan
                elif batch_first_y_pred[batch_idx][cond_idx] == 3:
                    batch_first_y_pred[batch_idx][cond_idx] = UNCERTAIN
                elif batch_first_y_pred[batch_idx][cond_idx] == 2:
                    batch_first_y_pred[batch_idx][cond_idx] = NEGATIVE
                elif batch_first_y_pred[batch_idx][cond_idx] == 1:
                    batch_first_y_pred[batch_idx][cond_idx] = POSITIVE

        return batch_first_y_pred

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

    def generate_complacency(
        self, finding,
        finding_label_df, fd_sent_label_df,
        chexbert_model, chexbert_tokenizer, batch_size, device,
        num_of_insert_sent
        ):

        generated_errors = {
            'complacency': []
        }

        fd_features = self._get_features(finding_label_df, finding)

        # Exclude "No Finding" label
        _fd_features = fd_features[:NO_FINDING_IDX]

        neg_cond_idxs = np.where(_fd_features == NEGATIVE)[0]
        neg_conds = [self.condition_list[idx] for idx in neg_cond_idxs]

        __fd_features = np.array(list(map(lambda x: None if np.isnan(x) else x, _fd_features)), dtype=np.float64)
        nan_cond_idxs = np.where(__fd_features == None)[0]
        nan_conds = [self.condition_list[idx] for idx in nan_cond_idxs]

        # if there is no neg or nan label, cannot generate complacency error
        if len(neg_conds) == 0 and len(nan_conds) == 0:
            return generated_errors
        
        conds = neg_conds + nan_conds
        random_idx = random.randrange(len(conds))
        target_cond = conds[random_idx]
        if target_cond in neg_conds:
            target_cond_label = NEGATIVE
        elif target_cond in nan_conds:
            target_cond_label = np.nan
        else:
            raise NotImplementedError

        target_opp_fd_sent_dict = {}
        fd_sents = sent_tokenize(finding)
        for sent_id in range(len(fd_sents)):

            fd_sent_features = self._get_features(fd_sent_label_df, fd_sents[sent_id])
            # Exclude "No Finding" label
            _fd_sent_features = fd_sent_features[:NO_FINDING_IDX]

            target_cond_idx = CONDITIONS_DICT[target_cond]
            if np.isnan(target_cond_label) and not np.isnan(_fd_sent_features[target_cond_idx]):
                continue
            if target_cond_label == NEGATIVE and _fd_sent_features[target_cond_idx] != target_cond_label:
                continue

            target_opp_idxes = fd_sent_label_df[
                fd_sent_label_df[target_cond] == POSITIVE
            ].index.tolist()

            random.shuffle(target_opp_idxes)
            sample_target_opp_idxes = target_opp_idxes[:20]
            sample_target_opp_idxes.sort(reverse=False)

            tmp_df = fd_sent_label_df.iloc[sample_target_opp_idxes]
            target_opp_fd_sent_list = tmp_df.loc[:, REPORTS].values.tolist()
            # target_opp_fd_sent_features_list = tmp_df.loc[:, CONDITIONS].values.tolist()
            target_opp_fd_sent_dict[sent_id] = target_opp_fd_sent_list

        insert_fd_sent_list = []
        for _, fd_sent_list in target_opp_fd_sent_dict.items():
            insert_fd_sent_list += fd_sent_list

        if len(insert_fd_sent_list) == 0:
            return generated_errors

        insert_fd_sent_list = list(set(insert_fd_sent_list))

        error_fd_cand_list = []
        for _ in range(batch_size):
            error_fd_cand = self.delete_and_insert_finding_sents(
                target_cond_label, target_opp_fd_sent_dict,
                insert_fd_sent_list,
                finding,
                num_of_insert_sent
            )
            
            error_fd_cand_list.append(error_fd_cand)

        if len(error_fd_cand_list) == 0:
            return generated_errors

        error_fd_features_list = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            error_fd_cand_list
        )
        for error_fd_features in error_fd_features_list:
            self._convert_tree_structure(error_fd_features)

        # Exclude "No Finding"
        _error_fd_features_list = [error_fd_features[:NO_FINDING_IDX] for error_fd_features in error_fd_features_list]

        complacency_list = []
        for i in range(len(_error_fd_features_list)):
            if self._is_complacency(target_cond, _error_fd_features_list[i]):
                #complacency_list.append(error_cand_list[i])
                complacency_list += [(error_fd_cand_list[i], _error_fd_features_list[i])]

        if len(complacency_list) > 0:
            random_idx = random.randrange(len(complacency_list))
            generated_errors['complacency'] += [complacency_list[random_idx]]
        
        return generated_errors

    def delete_and_insert_finding_sents(
        self, target_cond_label, target_opp_fd_sent_dict, 
        insert_fd_sent_list, 
        finding,
        num_of_insert_sent
    ):

        fd_sents = sent_tokenize(finding)

        error_fd_sents = []
        # if target label of finding_sentence is negative, delete that sentence
        # else remain that sentence 
        for sent_id in range(len(fd_sents)):
            if target_cond_label == NEGATIVE and sent_id in target_opp_fd_sent_dict:
                continue
            else:
                error_fd_sents.append(fd_sents[sent_id])

        # insert sentence which has opposite label with target label into finding
        if len(error_fd_sents) > 0:
            for _ in range(num_of_insert_sent):
                insert_pos = random.randrange(len(error_fd_sents))
                insert_fd_sent = insert_fd_sent_list[random.randrange(len(insert_fd_sent_list))]

                error_fd_sents.insert(insert_pos, insert_fd_sent)

        error_finding = " ".join(error_fd_sents)

        return error_finding

    def _is_complacency(self, target_cond, error_fd_features):
        target_cond_idx = CONDITIONS_DICT[target_cond]
        if error_fd_features[target_cond_idx] == POSITIVE:
            return True
        else:
            return False

    def generate_faulty_reasoning(
        self, finding: str, 
        finding_label_df, fd_sent_label_df,
        chexbert_model, chexbert_tokenizer, batch_size, device,
        fr1_prob=.5, fr2_prob=.5, swap_prob=.5
    ):

        generated_errors = {
            'faulty_reasoning_1': [],
            'faulty_reasoning_2': [],
        }

        fd_features = self._get_features(finding_label_df, finding)

        if np.random.choice([True, False], 1, p=[fr1_prob, 1-fr1_prob]).item():
            faulty_reasoning_1_list = self._get_faulty_reasoning_1(
                finding, fd_features, finding_label_df
            )
            if len(faulty_reasoning_1_list) > 0:
                random_idx = random.randrange(len(faulty_reasoning_1_list))
                generated_errors['faulty_reasoning_1'] += [faulty_reasoning_1_list[random_idx]]

        if np.random.choice([True, False], 1, p=[fr2_prob, 1-fr2_prob]).item():
            faulty_reasoning_2_list = self._get_faulty_reasoning_2(
                finding, fd_features, fd_sent_label_df, swap_prob,
                chexbert_model, chexbert_tokenizer, batch_size, device
            )
            if len(faulty_reasoning_2_list) > 0:
                random_idx = random.randrange(len(faulty_reasoning_2_list))
                generated_errors['faulty_reasoning_2'] += [faulty_reasoning_2_list[random_idx]]

        return generated_errors

    def _get_faulty_reasoning_1(
        self, finding: str, fd_features, finding_label_df
        ):

        # if there is no finding, cannot generate faulty_reasoning_1 error
        if fd_features[NO_FINDING_IDX] == POSITIVE:
            return []
        # Exclude "No Finding" class
        _fd_features = fd_features[:NO_FINDING_IDX]

        pos_cond_idxs = np.where(_fd_features == POSITIVE)[0]
        pos_conds = [self.condition_list[idx] for idx in pos_cond_idxs]
        #pos_conds = list(map(lambda x: self.condition_list[x], pos_cond_idxs))

        neg_cond_idxs = np.where(_fd_features == NEGATIVE)[0]
        neg_conds = [self.condition_list[idx] for idx in neg_cond_idxs]
        #neg_conds = list(map(lambda x: self.condition_list[x], neg_cond_idxs))

        pos_opp_fd_idxes = []
        for pos_cond in pos_conds:
            curr_pos_opp_fd_idxes = finding_label_df[
                finding_label_df[pos_cond] == NEGATIVE
            ].index
            pos_opp_fd_idxes += list(curr_pos_opp_fd_idxes)

        neg_opp_fd_idxes = []
        for neg_cond in neg_conds:
            curr_neg_opp_fd_idxes = finding_label_df[
                finding_label_df[neg_cond] == POSITIVE
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
        sample_opp_fd_idxes = opp_fd_idxes[:20]
        sample_opp_fd_idxes.sort(reverse=False)

        tmp_df = finding_label_df.iloc[sample_opp_fd_idxes]
        swap_findings = tmp_df.loc[:, REPORTS].values.tolist()
        swap_fd_features_list = tmp_df.loc[:, CONDITIONS].values.tolist()
        _swap_fd_features_list = [sf[:NO_FINDING_IDX] for sf in swap_fd_features_list]
#        random_idx = random.randrange(len(swap_findings))
        
        faulty_reasoning_1_list = []
        for i in range(len(swap_findings)):
            if self._is_faulty_reasoning(_fd_features, _swap_fd_features_list[i]):
                faulty_reasoning_1_list += [(swap_findings[i], _swap_fd_features_list[i])]
        
        return faulty_reasoning_1_list

    def _get_faulty_reasoning_2(
        self, finding: str, fd_features, 
        fd_sent_label_df,
        swap_prob,
        chexbert_model, chexbert_tokenizer, batch_size, device        
        ):


        # if there is no finding, cannot generate faulty_reasoning_2 error
        if fd_features[NO_FINDING_IDX] == POSITIVE:
            return []

        # Exclude "No Finding" label
        _fd_features = fd_features[:NO_FINDING_IDX]

        fd_pos_cond_idxs = np.where(_fd_features == POSITIVE)[0]
        fd_pos_conds = [self.condition_list[idx] for idx in fd_pos_cond_idxs]

        fd_neg_cond_idxs = np.where(_fd_features == NEGATIVE)[0]
        fd_neg_conds = [self.condition_list[idx] for idx in fd_neg_cond_idxs]

        swap_fd_sent_dict = {}
        fd_sents = sent_tokenize(finding)
        for sent_id in range(len(fd_sents)):

            fd_sent_features = self._get_features(fd_sent_label_df, fd_sents[sent_id])
            # Exclude "No Finding"
            _fd_sent_features = fd_sent_features[:NO_FINDING_IDX]

            fd_sent_pos_cond_idxs = np.where(_fd_sent_features == POSITIVE)[0]
            fd_sent_pos_conds = [self.condition_list[idx] for idx in fd_sent_pos_cond_idxs]

            fd_sent_neg_cond_idxs = np.where(_fd_sent_features == NEGATIVE)[0]
            fd_sent_neg_conds = [self.condition_list[idx] for idx in fd_sent_neg_cond_idxs]

            # Select positive or negative conditions in which both finding and finding_sent share at the same time
            pos_conds = [fd_sent_pos_cond for fd_sent_pos_cond in fd_sent_pos_conds if fd_sent_pos_cond in fd_pos_conds]
            neg_conds = [fd_sent_neg_cond for fd_sent_neg_cond in fd_sent_neg_conds if fd_sent_neg_cond in fd_neg_conds]

            if len(pos_conds) == 0 and len(neg_conds) == 0:
                continue


            pos_opp_fd_sent_idxes = []
            for pos_cond in pos_conds:
                curr_pos_opp_fd_sent_idxes = fd_sent_label_df[
                    fd_sent_label_df[pos_cond] == NEGATIVE
                ].index
                pos_opp_fd_sent_idxes += list(curr_pos_opp_fd_sent_idxes)

            neg_opp_fd_sent_idxes = []
            for neg_cond in neg_conds:
                curr_neg_opp_fd_sent_idxes = fd_sent_label_df[
                    fd_sent_label_df[neg_cond] == POSITIVE
                ].index
                neg_opp_fd_sent_idxes += list(curr_neg_opp_fd_sent_idxes)


            opp_fd_sent_idxes = pos_opp_fd_sent_idxes + neg_opp_fd_sent_idxes
            opp_fd_sent_idxes = list(set(opp_fd_sent_idxes))
            if len(opp_fd_sent_idxes) > 0:
                random.shuffle(opp_fd_sent_idxes)
                sample_opp_fd_sent_idxes = opp_fd_sent_idxes[:10]
                sample_opp_fd_sent_idxes.sort(reverse=False)

                tmp_df = fd_sent_label_df.iloc[sample_opp_fd_sent_idxes]
                swap_fd_sent_list = tmp_df.loc[:, REPORTS].values.tolist()
                # swap_fd_features_list = tmp_df.loc[:, CONDITIONS].values.tolist()
                swap_fd_sent_dict[sent_id] = swap_fd_sent_list

        error_fd_cand_list = []
        for _ in range(batch_size):
            error_fd_cand = self._swap_top_k_finding_sents(
                swap_fd_sent_dict,
                finding,
                swap_prob
            )
            if len(error_fd_cand) > 10:
                error_fd_cand_list.append(error_fd_cand)

        error_fd_features_list = self._chexbert_forward(
            chexbert_model, chexbert_tokenizer, batch_size, device,
            error_fd_cand_list
        )
        for error_fd_features in error_fd_features_list:
            self._convert_tree_structure(error_fd_features)

        # Exclude "No Finding"
        _error_fd_features_list = [error_features[:NO_FINDING_IDX] for error_features in error_fd_features_list]

        faulty_reasoning_2_list = []
        for i in range(len(_error_fd_features_list)):
            if self._is_faulty_reasoning(_fd_features, _error_fd_features_list[i]):
                #faulty_reasoning_2_list.append(error_cand_list[i])
                faulty_reasoning_2_list += [(error_fd_cand_list[i], _error_fd_features_list[i])]

        return faulty_reasoning_2_list


    def _swap_top_k_finding_sents(
        self, swap_fd_sent_dict, finding, swap_prob
    ):

        fd_sents = sent_tokenize(finding)

        error_fd_sents = []       
        swap_count = 0
        for sent_id in range(len(fd_sents)):
            if sent_id in swap_fd_sent_dict:
                is_swap = np.random.choice([True, False], 1, p=[swap_prob, 1-swap_prob]).item()
                if is_swap:
                    swap_fd_sent_list = swap_fd_sent_dict[sent_id]
                    error_fd_sents.append(
                        swap_fd_sent_list[random.randrange(len(swap_fd_sent_list))]
                    )
                    swap_count += 1
                else:
                    error_fd_sents.append(fd_sents[sent_id])   
            else:
                error_fd_sents.append(fd_sents[sent_id])

        error_finding = " ".join(error_fd_sents)

        return error_finding

    def _is_faulty_reasoning(self, _fd_features, _error_fd_features):
        
        def get_cluster_idx(cond_idx):
            cluster_idx = None
            for idx, conds_of_cluster in enumerate(INTERCHANGABLE_CONDITIONS_CLUSTER):
                cond_idxs_of_cluster = [CONDITIONS_DICT[cond] for cond in conds_of_cluster]

                if cond_idx in cond_idxs_of_cluster:
                    cluster_idx = idx

            return cluster_idx

        pos_to_neg_cond_idx_list = []
        neg_to_pos_cond_idx_list = []
        for cond_idx in range(len(_fd_features)):
            if _fd_features[cond_idx] == POSITIVE and _error_fd_features[cond_idx] == NEGATIVE:
                pos_to_neg_cond_idx_list += [cond_idx]
            if _fd_features[cond_idx] == NEGATIVE and _error_fd_features[cond_idx] == POSITIVE:
                neg_to_pos_cond_idx_list += [cond_idx]

        # both A and B abnormality should be changed at the same time
        if len(pos_to_neg_cond_idx_list) == 0 or len(neg_to_pos_cond_idx_list) == 0:
            return False

        is_fr = False
        for ptn_cond_idx in pos_to_neg_cond_idx_list:
            for ntp_cond_idx in neg_to_pos_cond_idx_list:
                ptn_cluster_idx = get_cluster_idx(ptn_cond_idx)
                ntp_cluster_idx = get_cluster_idx(ntp_cond_idx)
                
                if ptn_cluster_idx is not None and ntp_cluster_idx is not None:
                    # if A and B abnormality are not in the same cluster, 
                    # both cannot be interchangable, so faulty reasoning!
                    if ptn_cluster_idx != ntp_cluster_idx:
                        is_fr = True
                else:
                    is_fr = True

        return is_fr

    def generate_random_swap(
        self, finding: str, finding_label_df
        ):

        generated_errors = {
            'random_swap': []
        }

        fd_features = self._get_features(finding_label_df, finding)
        # Exclude "No Finding" class
        _fd_features = fd_features[:NO_FINDING_IDX]

        loop_step = 0
        random_idxes_set = set()
        while loop_step < 100 or len(random_idxes_set) < 50:
            random_idx = random.randrange(finding_label_df.shape[0])
            random_idxes_set.add(random_idx)
            loop_step += 1
        random_idxes = list(random_idxes_set)
        random_idxes.sort(reverse=False)

        tmp_df = finding_label_df.iloc[random_idxes]
        rand_swap_fd_findings = tmp_df.loc[:, REPORTS].values.tolist()
        rand_swap_fd_features_list = tmp_df.loc[:, CONDITIONS].values.tolist()
        _rand_swap_fd_features_list = [rsf[:NO_FINDING_IDX] for rsf in rand_swap_fd_features_list]

        random_swap_list = []
        for i in range(len(random_idxes)):
            if self._is_random_swap(_fd_features, _rand_swap_fd_features_list[i]):
                random_swap_list += [(rand_swap_fd_findings[i], _rand_swap_fd_features_list[i])]

        if len(random_swap_list) > 0:
            random_idx = random.randrange(len(random_swap_list))
            generated_errors['random_swap'] += [random_swap_list[random_idx]]

        return generated_errors

    def _is_random_swap(self, _fd_features, _rand_swap_fd_features):

        # to compare ndarray, need to convert np.nan to None
        __fd_features = np.array(list(map(lambda x: None if np.isnan(x) else x, _fd_features)), dtype=np.float64)
        __rand_swap_fd_features = np.array(list(map(lambda x: None if np.isnan(x) else x, _rand_swap_fd_features)), dtype=np.float64)

        return not (__fd_features == __rand_swap_fd_features).all()

    def _get_idx_of_positive_label(self, features_list):
        pos_idxs_list = []
        for features in features_list:
            pos_idxs = [idx for idx in range(len(features)) if features[idx] == POSITIVE]
            pos_idxs_list.append(pos_idxs)
        
        return pos_idxs_list

    def _get_features(self, df, text):

        features = df[
            df[REPORTS] == text
        ].values[0][1:]

        return features