"""
import os, sys
currentdir = os.path.abspath('')
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
"""

from config import *
from main_package.bkts import bkt
from main_package.utils import csv_to_fixed_params
from sklearn.metrics import roc_auc_score
from math import isclose
import pytest


def test_bkt_on_piech_data():
    #ARRANGE
    fixed_params = csv_to_fixed_params("piech/test_df_format")
    params = [0.5504063365919529, 0.3, 0.0999999975569658, 0.06302321901515896] #p_L0, p_G, p_S, p_T

    #ACT
    result = bkt(params, fixed_params, evaluate=True)
    
    #ASSERT
    #the numbers are confirmed by my first_attempt implementation
    squared_sum_residuals = result[1]
    assert squared_sum_residuals == 21684.4644963965

    correct, prediction, _  = result[0]
    assert isclose(roc_auc_score(correct, prediction), 0.7295786468978473, abs_tol=0.001) # 0.73 is a reference value from a paper how deep is KT

@pytest.mark.parametrize("params,squared_sum_residuals",
    [
        #[p_L0, p_G, p_S, p_T], value from an excel calculation
        ([0.3, 0.25, 0.2, 0.1], 405.4794423),
        ([0.5, 0.3, 0.1, 0.08], 397.1507716),
        ([0.53, 0.31, 0.11, 0.105], 392.7054479),
        ([0.0261, 0.06404, 0.60008, 0.05054], 364.00258750299673), #this row is from pyBKT
    ]
)
def test_on_edx_data(params, squared_sum_residuals):
    #ARRANGE
    fixed_params = csv_to_fixed_params("edx/Asgn4-dataset-filtered")

    #ACT
    result = bkt(params, fixed_params, evaluate=True)
    
    #ASSERT
    squared_sum_residuals = result[1]
    assert isclose(squared_sum_residuals, squared_sum_residuals)

@pytest.mark.parametrize("params,squared_sum_residuals",
    [
        #[p_L0, p_G, p_S, p_T], value from an excel calculation
        ([0.4, 0.2, 0.1, 0.2], 2.886432),
        ([0.5, 0.3, 0.09, 0.1], 2.858508348),
    ]
)
def test_on_my_sample_data(params, squared_sum_residuals):
    #ARRANGE
    fixed_params = csv_to_fixed_params("my_sample_data")

    #ACT
    result = bkt(params, fixed_params, evaluate=True)
    
    #ASSERT
    squared_sum_residuals = result[1]
    assert isclose(squared_sum_residuals, squared_sum_residuals)