from typing import Dict, Iterable, List, Optional, Union


def bkt(
        params: List[float],
        fixed_params: List[Iterable],
        evaluate=False,
        split_skill: Optional[str]='_'
    ):
    p_L0, p_G, p_S, p_T = params
    user_id, skill_id, correct = fixed_params

    assert len(user_id) == len(skill_id) == len(correct)
    n = len(skill_id)

    user_skill_lookup: Dict[str, Dict[str, float]] = {}

    p_prior, p_prior_cond, p_posterior = 0, 0, 0 #PLn-1, PLn-1|answer, PLn
    correct_results, likelihoods = [], []
    squared_sum_residuals = 0
    for i in range(n):
        if not user_id[i] in user_skill_lookup:
            user_skill_lookup[user_id[i]] = {}

        skills = [skill_id[i]] if split_skill is None else skill_id[i].split(split_skill)
        for skill in skills:
            if skill in user_skill_lookup[user_id[i]]:
                p_prior = user_skill_lookup[user_id[i]][skill]
            else:
                p_prior = p_L0

            likelihood_correct = p_prior * (1 - p_S) + (1 - p_prior) * p_G
            squared_sum_residuals += (correct[i] - likelihood_correct)**2

            if evaluate:
                likelihoods.append(likelihood_correct)
                correct_results.append(correct[i])

            if correct[i]:
                p_did_not_slip_and_knew = (1 - p_S) * p_prior
                p_guessed_and_did_not_know = p_G * (1 - p_prior)
                p_prior_cond = p_did_not_slip_and_knew / (p_did_not_slip_and_knew + p_guessed_and_did_not_know)
            else:
                p_slipped_and_knew = p_S * p_prior
                p_did_not_guess_and_did_not_know = (1 - p_G) * (1 - p_prior)
                p_prior_cond = p_slipped_and_knew / (p_slipped_and_knew + p_did_not_guess_and_did_not_know)

            p_posterior = p_prior_cond + p_T * (1 - p_prior_cond)
            user_skill_lookup[user_id[i]][skill] = p_posterior
    
    return (correct_results, likelihoods, user_skill_lookup), squared_sum_residuals


def bkt_forget(
        params: List[float],
        fixed_params: List[Iterable],
        evaluate=False,
        split_skill: Optional[str]='_'
    ):
    p_L0, p_G, p_S, p_T, p_F = params
    user_id, skill_id, correct = fixed_params

    assert len(user_id) == len(skill_id) == len(correct)
    n = len(skill_id)
    
    user_skill_lookup: Dict[str, Dict[str, Union[int, Dict[str, float]]]] = {}
    previous_p_posterior = "p_prior"
    last_seen = "last_seen"
    user_attempt_counts = "attempt_counts"
    
    p_prior, p_prior_cond, p_posterior = 0, 0, 0 #PLn-1, PLn-1|answer, PLn
    priors, correct_results = [], []
    squared_sum_residuals = 0
    for i in range(n):
        if not user_id[i] in user_skill_lookup:
            user_skill_lookup[user_id[i]] = {
                user_attempt_counts: 0
            }
        user_dict = user_skill_lookup[user_id[i]]
        user_dict[user_attempt_counts] += 1

        skills = [skill_id[i]] if split_skill is None else skill_id[i].split(split_skill)
        for skill in skills:
            if skill in user_dict:
                p_did_not_forget = (1 - p_F)**(user_dict[user_attempt_counts] - user_dict[skill][last_seen])
                # print(f'attempt counts: {user_dict[user_attempt_counts]}, last_seen: {user_dict[skill][last_seen]}')
                # print(f'p_did_not_forget: {p_did_not_forget}')
                p_prior = user_dict[skill][previous_p_posterior] * p_did_not_forget
            else:
                p_prior = p_L0

            likelihood_correct = p_prior * (1 - p_S) + (1 - p_prior) * p_G
            squared_sum_residuals += (correct[i] - likelihood_correct)**2

            if evaluate:
                priors.append(p_prior)
                correct_results.append(correct[i])

            if correct[i]:
                p_did_not_slip_and_knew = (1 - p_S) * p_prior
                p_guessed_and_did_not_know = p_G * (1 - p_prior)
                p_prior_cond = p_did_not_slip_and_knew / (p_did_not_slip_and_knew + p_guessed_and_did_not_know)
            else:
                p_slipped_and_knew = p_S * p_prior
                p_did_not_guess_and_did_not_know = (1 - p_G) * (1 - p_prior)
                p_prior_cond = p_slipped_and_knew / (p_slipped_and_knew + p_did_not_guess_and_did_not_know)
            

            p_posterior = p_prior_cond + p_T * (1 - p_prior_cond)
            user_dict[skill] = {
                previous_p_posterior: p_posterior,
                last_seen: user_dict[user_attempt_counts],
            }
            # print(f'p_prior: {p_prior}, likelihood_correct: {likelihood_correct}, p_prior_cond: {p_prior_cond}, p_posterior: {p_posterior}')
    
    return (correct_results, priors, user_skill_lookup), squared_sum_residuals