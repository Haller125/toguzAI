import pandas as pd
import ast

df = pd.read_csv("ultimate_porn.csv")

# white_features: [[row.white[n], 1 if row.tuzdeks[0] = n else 0, row.kaznas[0]] for n in range of 9]
# black_features: [[row.black[n], 1 if row.tuzdeks[1] = n else 0, row.kaznas[1]] for n in range of 9]
def extract_white_features(row):
    return [[row.white[n], 1 if row.tuzdeks[0] == n else 0, row.kaznas[0]] for n in range(9)]

def extract_black_features(row):
    return [[row.black[n], 1 if row.tuzdeks[1] == n else 0, row.kaznas[1]] for n in range(9)]

if __name__ == "__main__":
    df["white"] = df["white"].apply(ast.literal_eval)
    df["black"] = df["black"].apply(ast.literal_eval)
    df["kaznas"] = df["kaznas"].apply(ast.literal_eval)
    df["tuzdeks"] = df["tuzdeks"].apply(ast.literal_eval)

    df["white_features"] = df.apply(extract_white_features, axis=1)
    df["black_features"] = df.apply(extract_black_features, axis=1)
    df["stm"] = df["stm"].apply(lambda x: 0 if x == "False" else 1)
    df.to_csv("ultimate_porn_sss.csv")
