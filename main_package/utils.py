from typing import Iterable, List, Optional, Tuple
import pandas as pd
from scipy.optimize import minimize
import time
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import copy

my_dtype = {
    "user_id": str,
    "skill_id": str,
    "correct": bool
}

def data_path_to_abs_path_old(path_in_data_folder: str, file_type: str = "csv") -> str:
    currentdir = os.path.dirname(__file__)
    parentdir = os.path.dirname(currentdir)
    return parentdir + "/data/" + path_in_data_folder + "." + file_type


def data_path_to_abs_path(path_in_data_folder: str) -> str:
    currentdir = os.path.dirname(__file__)
    parentdir = os.path.dirname(currentdir)
    return parentdir + "/data/" + path_in_data_folder


def csv_to_fixed_params(path_in_data_folder: str) -> List[Iterable]:
    df = pd.read_csv(data_path_to_abs_path_old(path_in_data_folder), dtype=my_dtype)
    return [
        df["user_id"].to_numpy(dtype=str),
        df["skill_id"].to_numpy(dtype=str),
        df["correct"].to_numpy(dtype=bool),
    ]


def csv_to_df_fixed_params_old(path_in_data_folder: str) -> pd.DataFrame:
    return pd.read_csv(data_path_to_abs_path_old(path_in_data_folder), dtype=my_dtype)[["user_id", "skill_id", "correct"]]


def csv_to_df_fixed_params(path_in_data_folder: str) -> pd.DataFrame:
    return pd.read_csv(data_path_to_abs_path(path_in_data_folder), dtype=my_dtype)[["user_id", "skill_id", "correct"]]


dx = 0.0001
method = "L-BFGS-B" # default for bounded: "L-BFGS-B", other: "Powell", "TNC", "Nelder-Mead"
def minimize_bkt(
    bkt_implementation,
    fixed_params: List[Iterable],
    bkt_type: str = "standard",
    initial_guess: Optional[List[float]] = None, 
    bounds: Optional[List[Tuple[float, float]]] = None, 
    method=method
    ) -> List[float]:
        def bkt_minimize(params, fixed_params):
            _, squared_sum_residuals = bkt_implementation(params, fixed_params)
            return squared_sum_residuals
        
        if bkt_type == "standard":
            if initial_guess == None:
                initial_guess = [0.4, 0.2, 0.08, 0.1]
            if bounds == None:
                bounds = [(dx, 1-dx), (dx, 0.3), (dx, 0.1), (dx, 1-dx)]
        elif bkt_type == "forget":
            if initial_guess == None:
                initial_guess = [0.4, 0.2, 0.08, 0.1, 0.01]
            if bounds == None:
                bounds = [(dx, 1-dx), (dx, 0.3), (dx, 0.1), (dx, 1-dx), (dx, 1-dx)]
        else:
            raise Exception(f"bkt_type: {bkt_type} not supported")

        start_time = time.time()
        result = minimize(bkt_minimize, initial_guess, args=fixed_params, bounds=bounds, method=method)
        end_time = time.time()
        diff = end_time - start_time
        print(f"training time was: {int(diff // 60)}m {round(diff % 60)}s")
        print(f"number of records: {len(fixed_params[0])}")
        print(f"optimized by: {method}")
        if len(result.x) == 4:
            #standard bkt
            print(f"p_L0 = {result.x[0]}\np_G = {result.x[1]}\np_S = {result.x[2]}\np_T = {result.x[3]}")
        else:
            assert len(result.x) == 5
            print(f"p_L0 = {result.x[0]}\np_G = {result.x[1]}\np_S = {result.x[2]}\np_T = {result.x[3]}\np_F = {result.x[4]}")

        print(result)
        return result.x.tolist()


def evaluate_bkt(
    bkt_implementation,
    fixed_params: List[Iterable], 
    optimal_values: List[float],
):
    result = bkt_implementation(optimal_values, fixed_params, evaluate=True)
    correct, prediction, _user_skill_lookup  = result[0]
    rounded_prediction = [round(p) for p in prediction]
    squared_sum_residuals = result[1]
    print(f"number of records: {len(prediction)}") 
    print(f"auc: {roc_auc_score(correct, prediction)}")
    print(f"accuracy: {accuracy_score(correct, rounded_prediction)}")
    print(f"squared sum of residuals: {squared_sum_residuals}")


def truncate_interaction_sequences(df: pd.DataFrame, max_interaction_len = 200):
    df = copy.deepcopy(df)
    for column_name in ['questions', 'concepts', 'responses', 'is_repeat']:
        if column_name in df.columns:
            df[column_name] = df[column_name].apply(lambda x: x[:max_interaction_len])
    return df
