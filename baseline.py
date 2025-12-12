import pandas as pd

df = pd.read_csv("labels.csv")

files = df["files"].tolist()

print(files)