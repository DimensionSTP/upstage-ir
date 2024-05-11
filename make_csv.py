import os
import glob as glob

import pandas as pd
from sklearn.model_selection import train_test_split


def read_file(articles_path, summaries_path, categories_list, encoding="ISO-8859-1"):
    articles = []
    summaries = []
    categories = []
    for category in categories_list:
        article_paths = glob.glob(
            os.path.join(articles_path, category, "*.txt"), recursive=True
        )
        summary_paths = glob.glob(
            os.path.join(summaries_path, category, "*.txt"), recursive=True
        )

        print(
            f"found {len(article_paths)} file in articles/{category} folder, {len(summary_paths)} file in summaries/{category}"
        )

        if len(article_paths) != len(summary_paths):
            print("number of files is not equal")
            return
        for file in range(len(article_paths)):
            categories.append(category)
            with open(article_paths[file], mode="r", encoding=encoding) as files:
                articles.append(files.read())

            with open(summary_paths[file], mode="r", encoding=encoding) as files:
                summaries.append(files.read())

    print(
        f"total {len(articles)} file in articles folder and {len(summaries)} files in summaries folder"
    )
    return articles, summaries, categories


if __name__ == "__main__":
    DATA_PATH = "/data/kaggle-bbc-news-summary/data"
    ARTICLES_PATH = f"{DATA_PATH}/BBC News Summary/News Articles"
    SUMARRIES_PATH = f"{DATA_PATH}/BBC News Summary/Summaries"
    categories_list = os.listdir(ARTICLES_PATH)

    articles, summaries, categories = read_file(
        ARTICLES_PATH, SUMARRIES_PATH, categories_list
    )

    data = pd.DataFrame(
        {"articles": articles, "summaries": summaries, "categories": categories}
    )
    csv_path = f"{DATA_PATH}/data.csv"
    data.to_csv(csv_path, index=False)
    train_data, test_data = train_test_split(
        data,
        test_size=0.1,
        random_state=2024,
        shuffle=True,
        stratify=data["categories"],
    )

    train_data.to_csv(f"{DATA_PATH}/train.csv", index=False)
    test_data.to_csv(f"{DATA_PATH}/test.csv", index=False)
