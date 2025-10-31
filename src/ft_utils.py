# src/ft_utils.py
import os, random
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

COLUMN_TYPES: Dict[str, str] = {
    'lbbb_paced': 'cat','age': 'num','gender': 'cat','height': 'num','weight': 'num',
    'hypertension': 'cat','diabetes': 'cat','hyperlipidemia': 'cat','nyha': 'cat','cld': 'cat',
    'egfr': 'num','cad': 'cat','pre_isch_stroke': 'cat','pad': 'cat','pre_valve_surgery': 'cat',
    'prior_mi': 'cat','prior_pci': 'cat','prior_cabg': 'cat','sts': 'num','euroscorei': 'num',
    'euroscoreii': 'num','LVEF': 'num','Vmax': 'num','mean_gradient': 'num','aortic_valve_area': 'num',
    'aortic_reg': 'cat','nsr': 'cat','lafb': 'cat','lpfb': 'cat','af': 'cat',
    'afl': 'cat','1st_avb': 'cat','other_avb': 'cat','pr_interval': 'num','qrs_interval': 'num',
    'qtc_interval': 'num','pre_balloon': 'cat','valve_type': 'cat','new_valve': 'cat','valve_size': 'num',
    'over_sizing': 'num','post_balloon': 'cat','bicuspid': 'cat','annu_short_dia': 'num','annu_long_dia': 'num',
    'annu_mean_dia': 'num','annu_area': 'num','annu_area_dia': 'num','annu_perimeter': 'num','annu_perimeter_dia': 'num',
    'sov_area': 'num','sinus_annu_ratio': 'num','ncc_dia': 'num','lcc_dia': 'num','rcc_dia': 'num',
    'stj_area': 'num','stj_annu_ratio': 'num','mean_dia': 'num','lcc_stj_height': 'num','lvot_area': 'num',
    'lvot_annu_ratio': 'num','lvot_short_dia': 'num','lvot_long_dia': 'num','total_ca': 'num','lcc_ca': 'num',
    'rcc_ca': 'num','ncc_ca': 'num','lca_height': 'num','rca_height': 'num','new_lbbb': 'cat'
}

def seed_everything(seed=21):
    random.seed(seed); np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def convert_column_types(df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for c, t in column_types.items():
        if c in df.columns:
            if t == 'num':
                df[c] = df[c].astype('float32')
            else:
                df[c] = df[c].astype('int64')
    return df

def preprocess(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[int], List[str]]:
    df = df_raw.copy()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(str).astype(int).values.reshape(-1,1)

    cat_cols = X.select_dtypes(['int64']).columns
    num_cols = X.select_dtypes(['float32']).columns

    if len(num_cols) > 0:
        ss = StandardScaler()
        X.loc[:, num_cols] = ss.fit_transform(X.loc[:, num_cols])

    cat_cardinalities: List[int] = []
    for col in cat_cols:
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X.loc[:, col])
        cat_cardinalities.append(X[col].nunique())

    return X, y.astype(np.float32), cat_cardinalities, list(df.columns[:-1])
