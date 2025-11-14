import pandas as pd

def load_mxmh_data(path="mxmh survey results2.csv"):
    df = pd.read_csv(path)
    df = df[["Anxiety", "Age", "Hours per day", "BPM"]]
    df = df.rename(columns={"Hours per day": "Duration"})

    return df
if __name__ == "__main__":
    df = load_mxmh_data()
    print(df.describe())
