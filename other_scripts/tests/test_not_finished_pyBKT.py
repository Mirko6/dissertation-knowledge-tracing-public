from pyBKT.models import Model
import pandas as pd
import numpy as np
from main_package.bkts import bkt_forget
from main_package.utils import csv_to_fixed_params, data_path_to_abs_path_old
import pytest


def generate_pyBKT_bkt_forget_test_cases():
    np.seterr(divide='ignore', invalid='ignore') #pyBKT raises some warnings that I should ignore
    defaults = {'skill_name': 'skill_id'}
    
    model = Model(seed = 42, num_fits = 1)
    df = pd.read_csv(data_path_to_abs_path_old("edx/Asgn4-dataset-filtered"))
    model.fit(data = df.sample(50, random_state=42), forgets=True, defaults=defaults)
    model_params = model.params()
    print(model_params)
    p_L0 = model_params.loc["VALUING-CAT-FEATURES"]["value"]["prior"]["default"]
    p_G = model_params.loc["VALUING-CAT-FEATURES"]["value"]["guesses"]["default"]
    p_S = model_params.loc["VALUING-CAT-FEATURES"]["value"]["slips"]["default"]
    p_T = model_params.loc["VALUING-CAT-FEATURES"]["value"]["learns"]["default"]
    p_F = model_params.loc["VALUING-CAT-FEATURES"]["value"]["forgets"]["default"]
    
    auc = model.evaluate(data = df, metric = 'auc')

    row = [p_L0, p_G, p_S, p_T, p_F, auc]

def _provide_values_for_custom_params_pyBKT(params, max_skill: int):
    p_L0, p_G, p_S, p_T, p_F = params
    return {
        str(i): {
            "prior": p_L0,
            "learns": np.array([p_T]),
            "guesses": np.array([p_G]),
            "slips": np.array([p_S]),
            "forgets": np.array([p_F]),
        } for i in range(max_skill)
    }

@pytest.mark.parametrize("params",
    [
        #[p_L0, p_G, p_S, p_T]
        [0.4, 0.2, 0.1, 0.2],
        [0.5, 0.3, 0.09, 0.1],
    ]
)
def test_bkt_forgetting(params):
    df_piech = pd.read_csv(data_path_to_abs_path_old("piech/sample_df_format"))
    max_skill = int(df_piech['skill_id'].max())
    defaults = {'skill_name': 'skill_id'}
    model = Model()
    model.coef_ = _provide_values_for_custom_params_pyBKT(params, max_skill)
    model.fit(data = df_piech.head(1), defaults=defaults, fixed=True)
    print(model.coef_ == _provide_values_for_custom_params_pyBKT(params, max_skill))
    rmse = model.evaluate(data = df_piech, metric = 'rmse')
    mse = rmse ** 2
    sum_squared_residuals = mse * len(df_piech)
    print(sum_squared_residuals)

    _, squared_sum_residuals = bkt_forget(params, csv_to_fixed_params("piech/sample_df_format"))
    print(squared_sum_residuals)

test_bkt_forgetting([0.4, 0.2, 0.1, 0.2, 0])