import time
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, accuracy_score
from main_package.bkt_pyKT import bkt_to_minimize, get_question_level_prediction



def train_bkt(
    df: pd.DataFrame,
    initial_guess: List[float] = [0.4, 0.2, 0.08, 0.1],
    ) -> Dict[str, Iterable[float]]:
        method = "L-BFGS-B"
        dx = 0.0001
        # standard BKT only
        assert len(initial_guess) == 4
        bounds = [(dx, 1-dx), (dx, 0.5), (dx, 0.5), (dx, 1-dx)]

        concepts = np.unique(np.concatenate(df['concepts'].reset_index(drop=True)))
        bkt_params = {}
        start_time = time.time()
        for concept in concepts:
            print(f"concept: {concept}")
            # filter df
            df['concepts_filter'] = df["concepts"].apply(lambda concepts: concepts==concept)
            df_filtered = pd.DataFrame(data={
                  "concepts": df.apply(lambda row: row['concepts'][row['concepts_filter']], axis=1),
                  "responses": df.apply(lambda row: row['responses'][row['concepts_filter']], axis=1),
                  "is_repeat": df.apply(lambda row: row['is_repeat'][row['concepts_filter']], axis=1),
            })

            #minimize for the concept
            result = minimize(bkt_to_minimize, initial_guess, args=df_filtered, bounds=bounds, method=method)
            bkt_params[concept] = result.x
        end_time = time.time()
        diff = end_time - start_time
        print(f"training time was: {int(diff // 60)}m {round(diff % 60)}s")
        print(f"optimized by: {method}")
        print(f"bkt_params")
        print(bkt_params)

        return bkt_params


def bkt_student_interactions(
        concepts: Iterable[str],
        responses: Iterable[int],
        is_repeat: Iterable[int],
        bkt_params_dict: Dict[str, Iterable[float]],
        default_params: Iterable[float],
    ) -> Iterable[float]:
    assert len(concepts) == len(responses)

    predictions = []
    concept_lookup_p_knowing: Dict[str, float] = {}
    concept_lookup_last_seen: Dict[str, int] = {}
    current_problem_number = 0
    for i in range(len(concepts)):
        bkt_params_list = bkt_params_dict.get(concepts[i], default_params)
        if len(bkt_params_list) == 4:
            p_init, p_guess, p_slip, p_transit = bkt_params_list
            p_forget = 0
        else:
            p_init, p_guess, p_slip, p_transit, p_forget = bkt_params_list

        if not is_repeat[i]:
            current_problem_number += 1

        # step 1 - predicting probability of a correct answer based on probability of knowing the concept
        p_knows_concept_prior = p_init
        if concepts[i] in concept_lookup_p_knowing:
            num_exercises_in_between = current_problem_number - concept_lookup_last_seen.get(concepts[i], 0)
            p_did_not_forget = (1 - p_forget)**num_exercises_in_between
            # print(f'current_problem_number: {current_problem_number}, last_seen: {concept_lookup_last_seen.get(concepts[i], 0)}')
            # print(f'p_did_not_forget: {p_did_not_forget}')
            p_knows_concept_prior = p_did_not_forget * concept_lookup_p_knowing[concepts[i]]

        p_correct_answer = p_knows_concept_prior * (1 - p_slip) + (1 - p_knows_concept_prior) * p_guess
        predictions.append(p_correct_answer)

        # step 2 - predicting probability of knowing the relevant concept given the correctness of the response
        p_knows_concept_conditioned = None
        if responses[i]:
            p_did_not_slip_and_knew = (1 - p_slip) * p_knows_concept_prior
            p_guessed_and_did_not_know = p_guess * (1 - p_knows_concept_prior)
            p_knows_concept_conditioned = p_did_not_slip_and_knew / (p_did_not_slip_and_knew + p_guessed_and_did_not_know)
        else:
            p_slipped_and_knew = p_slip * p_knows_concept_prior
            p_did_not_guess_and_did_not_know = (1 - p_guess) * (1 - p_knows_concept_prior)
            p_knows_concept_conditioned = p_slipped_and_knew / (p_slipped_and_knew + p_did_not_guess_and_did_not_know)

        p_knows_concept_posterior = p_knows_concept_conditioned + p_transit * (1 - p_knows_concept_conditioned)
        concept_lookup_p_knowing[concepts[i]] = p_knows_concept_posterior
        
        # print(f'p_knows_concept_prior: {p_knows_concept_prior}, p_correct_answer: {p_correct_answer}, p_knows_concept_conditioned: {p_knows_concept_conditioned}, p_knows_concept_posterior: {p_knows_concept_posterior}')
        
        concept_lookup_last_seen[concepts[i]] = current_problem_number

    return predictions

def bkt_all_interactions(bkt_params_dict: Dict[str, Iterable[float]], df: pd.DataFrame, default_params: Iterable[float]) -> pd.Series:
    return df.apply(
        lambda row: bkt_student_interactions(row['concepts'], row['responses'], row['is_repeat'], bkt_params_dict, default_params),
        axis=1,
    )


def evaluate_bkt(bkt_params_dict: Dict[str, Iterable[float]], df: pd.DataFrame, default_params: Iterable[float]) -> Tuple[float, float]:
    df['predictions'] = bkt_all_interactions(bkt_params_dict, df, default_params)
    predictions, responses = get_question_level_prediction(df)
    rounded_prediction = [round(p) for p in predictions]
    print(f"number of predicted questions: {len(predictions)}")
    auc = roc_auc_score(responses, predictions)
    print(f"auc: {auc}")
    accuracy = accuracy_score(responses, rounded_prediction)
    print(f"accuracy: {accuracy}")
    return auc, accuracy