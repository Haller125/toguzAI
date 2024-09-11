import pandas as pd
import numpy as np
import pickle
import json
from scipy.optimize import minimize
from scipy.special import expit

def create_ds_for_sigma():
    df = pd.read_parquet("ultimate_porn.parquet")
    df_1 = pd.DataFrame(columns=["white_features", "black_features", "stm", "eval", "result"])
    hs_lookup = {}
    with open('saved_dictionary.pkl', 'rb') as f:
        hs = pickle.load(f)
        for key in hs.keys():
            hs_lookup[key] = True
        f.close()
    for i in range(len(df)):
        row = df.iloc[i]
        sockets = np.concatenate([row['white'], row["black"]], axis=None) if not row["stm"] else np.concatenate([row["black"], row['white']], axis=None)
        store = [sockets.tolist(), row['tuzdeks'].tolist(), row['kaznas'].tolist()]
        store_key = json.dumps(store)

        if store_key not in hs_lookup:
            continue
        result = hs[store_key]
        if row['stm']:
            result = 1 - result
        new_row = pd.DataFrame({
            "white_features": [row['white']],
            "black_features": [row['black']],
            "stm": [row['stm']],
            "eval": [row["eval"]],
            "result": [result]
        })
        df_1 = pd.concat([df_1, new_row], ignore_index=True)
    df_1.to_csv("sigma1.csv")
    print(df_1.head())

def sigmoid(x, k=0.0025, x0=63.3998):
    return expit(k * (x - x0))
def loss_function(params, x, y_true):
    k, x0 = params
    y_pred = sigmoid(x, k, x0)
    return np.mean((y_pred - y_true) ** 2)

def transform_to_wdl_space():
    df = pd.read_parquet("flat_porn.parquet")
    df["eval"] = df["eval"].apply(sigmoid)
    print(df)
    df.to_parquet("dataset.parquet")

if __name__ == '__main__':
    # initial_params = np.array([0.1, 60])
    # x = pd.read_parquet("sigma1.parquet")["eval"]
    # y_true = pd.read_parquet("sigma1.parquet")["result"]
    # result = minimize(loss_function, initial_params, args=(x, y_true), method='Nelder-Mead')
    # k_opt, x0_opt = result.x
    #
    # print(f"Оптимальные параметры: k = {k_opt:.4f}, x0 = {x0_opt:.4f}")
    # create_ds_for_sigma()

    transform_to_wdl_space()
