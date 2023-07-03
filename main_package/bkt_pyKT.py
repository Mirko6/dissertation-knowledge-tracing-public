import time
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, accuracy_score

def bkt_student_interactions(
        concepts: Iterable[str],
        responses: Iterable[int],
        is_repeat: Iterable[int],
        p_init: float,
        p_guess: float,
        p_slip: float,
        p_transit: float,
        p_forget: float = 0,
    ) -> Iterable[float]:
    """
    BKT for one student.

    Args:
        concepts (Iterable[str]), responses (Iterable[int]), is_repeat (Iterable[int]):
            student's interaction sequences in order as they happened.
        p_init (float), p_guess (float), p_slip (float), p_transit (float), p_forget (float, optional), Defaults to 0:
            BKT parameters.

    Returns:
        Iterable[float]: predicted probabilites of student answering the respective question correctly
    """
    assert len(concepts) == len(responses)

    predictions = []
    concept_lookup_p_knowing: Dict[str, float] = {}
    concept_lookup_last_seen: Dict[str, int] = {}
    current_problem_number = 0
    for i in range(len(concepts)):
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


def bkt_all_interactions(bkt_params: Iterable[float], df: pd.DataFrame) -> pd.Series:
    """BKT for all students

    Args:
        bkt_params (Iterable[float]): p_init, p_guess, p_slip, p_transit, p_forget (optional, if not present it will be set to 0)
        df (pd.DataFrame): student interaction data with column_name: value_type
            concepts: Iterable[str]
            responses: Iterable[int] - 0-incorrect, 1-correct
            is_repeat: Iterable[int] - 0-not a repeat, 1-repeat

    Returns:
        pd.Series with values Iterable[float] - predicted probabilities of a correct answer
    """
    return df.apply(
        lambda row: bkt_student_interactions(row['concepts'], row['responses'], row['is_repeat'], *bkt_params),
        axis=1,
    )


def bkt_to_minimize(bkt_params: Iterable[float], df: pd.DataFrame) -> float:
    df['predictions'] = bkt_all_interactions(bkt_params, df)
    squared_sum_residuals = df.apply(
        lambda row: np.sum(np.square(row['predictions'] - row['responses'])),
        axis = 1
    )
    num_interactions = df['predictions'].apply(len)
    return np.average(squared_sum_residuals, weights=num_interactions)

 
def train_bkt(
    df: pd.DataFrame,
    initial_guess: List[float] = [0.4, 0.2, 0.08, 0.1, 0.01],
    ) -> List[float]:
        method = "L-BFGS-B"
        dx = 0.0001
        #bounds = [(dx, 1-dx), (dx, 0.3), (dx, 0.1), (dx, 1-dx), (dx, 1-dx)]
        bounds = [(dx, 1-dx), (dx, 0.5), (dx, 0.5), (dx, 1-dx), (dx, 1-dx)]

        start_time = time.time()
        result = minimize(bkt_to_minimize, initial_guess, args=df, bounds=bounds, method=method)
        end_time = time.time()
        diff = end_time - start_time
        print(f"training time was: {int(diff // 60)}m {round(diff % 60)}s")
        print(f"optimized by: {method}")
        print(f"p_L0 = {result.x[0]}\np_G = {result.x[1]}\np_S = {result.x[2]}\np_T = {result.x[3]}\np_F = {result.x[4]}")

        print(result)
        return result.x.tolist()


def get_question_level_prediction(df: pd.DataFrame) -> Tuple[List[float]]:
    """Late Fusion Average for BKT returns predictions, responses tuple"""
    predictions = []
    responses = []
    for _, row in df.iterrows():
        current_predictions = []
        for i in range(len(row['predictions'])):
            current_predictions.append(row['predictions'][i])
            if row['is_repeat'][i] == 0:
                predictions.append(np.mean(current_predictions))
                responses.append(row['responses'][i])
                current_predictions = []
    
        if len(current_predictions) > 0:
            predictions.append(np.mean(current_predictions))
            responses.append(row['responses'][-1])

    return predictions, responses

     
def evaluate_bkt(bkt_params: Iterable[float], df: pd.DataFrame) -> None:
    df['predictions'] = bkt_all_interactions(bkt_params, df)
    predictions, responses = get_question_level_prediction(df)
    rounded_prediction = [round(p) for p in predictions]
    print(f"number of predicted questions: {len(predictions)}") 
    print(f"auc: {roc_auc_score(responses, predictions)}")
    print(f"accuracy: {accuracy_score(responses, rounded_prediction)}")


def convert_df_strings_to_arrays(df: pd.DataFrame) -> None:
    for column_name in ['questions', 'concepts', 'responses', 'is_repeat']:
        dtype = None
        if column_name in ['responses', 'is_repeat']:
            dtype = np.int8
        df.loc[:, column_name] = df[column_name].apply(lambda x: np.array(x.split(','), dtype=dtype))