import pandas as pd


def add_empty_column(
    df_path: str,
    column_name: str,
):
    df = pd.read_csv(df_path)
    df[column_name] = " "
    df.to_csv(df_path, index=False)


if __name__ == "__main__":
    DATA_PATH = "/data/upstage-nlp/data"
    COLUMN_NAME = "summary"
    FILES = ["test", "sample_submission"]
    for file in FILES:
        add_empty_column(
            df_path=f"{DATA_PATH}/{file}.csv",
            column_name=COLUMN_NAME,
        )
