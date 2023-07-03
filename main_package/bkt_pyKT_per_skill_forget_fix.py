from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize


def bkt_student_interactions(
        concepts: Iterable[str],
        responses: Iterable[int],
        sequence_positions: Iterable[int],
        p_init: float,
        p_guess: float,
        p_slip: float,
        p_transit: float,
        p_forget: float,
    ) -> Iterable[float]:
    assert len(concepts) == len(responses) == len(sequence_positions)

    predictions = []
    concept_lookup_p_knowing: Dict[str, float] = {}
    for i in range(len(concepts)):
        # step 1 - predicting probability of a correct answer based on probability of knowing the concept
        p_knows_concept_prior = p_init
        if concepts[i] in concept_lookup_p_knowing:
            num_exercises_in_between = 0 if i==0 else sequence_positions[i] - sequence_positions[i-1]
            p_did_not_forget = (1 - p_forget)**num_exercises_in_between
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
        
    return predictions


def bkt_all_interactions(bkt_params: Iterable[float], df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda row: bkt_student_interactions(row['concepts'], row['responses'], row['sequence_positions'], *bkt_params),
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
    ) -> Dict[str, Iterable[float]]:
        method = "L-BFGS-B"
        dx = 0.0001
        # BKT with forgetting only
        assert len(initial_guess) == 5
        bounds = [(dx, 1-dx), (dx, 0.5), (dx, 0.5), (dx, 1-dx), (dx, 1-dx)]

        concepts = np.unique(np.concatenate(df['concepts'].reset_index(drop=True)))
        bkt_params = {}
        start_time = time.time()
        df['flipped_is_repeat_cum_sum'] = df['is_repeat'].apply(lambda is_repeat: np.cumsum(is_repeat*(-1) + 1))
        for concept in concepts:
            print(f"concept: {concept}")
            # filter df
            df['concepts_filter'] = df["concepts"].apply(lambda concepts: concepts==concept)

            df_filtered = pd.DataFrame(data={
                  "concepts": df.apply(lambda row: row['concepts'][row['concepts_filter']], axis=1),
                  "responses": df.apply(lambda row: row['responses'][row['concepts_filter']], axis=1),
                  "is_repeat": df.apply(lambda row: row['is_repeat'][row['concepts_filter']], axis=1),
                  "sequence_positions": df.apply(lambda row: row['flipped_is_repeat_cum_sum'][row['concepts_filter']], axis=1),
            })

            #minimize for the concept
            #bkt_to_minimize(initial_guess, df_filtered)
            result = minimize(bkt_to_minimize, initial_guess, args=df_filtered, bounds=bounds, method=method)
            bkt_params[concept] = result.x
        end_time = time.time()
        diff = end_time - start_time
        print(f"training time was: {int(diff // 60)}m {round(diff % 60)}s")
        print(f"optimized by: {method}")
        print(f"bkt_params")
        print(bkt_params)

        return bkt_params