from typing import Tuple, List

import kaggle
from pathlib import Path
import urllib.request
import pandas as pd
import os


def data_fetching() -> Tuple[str, str]:
    data_folder_path = "./data"
    raw_data_folder_path = os.path.join(data_folder_path, "raw")
    s_and_p_list_path = os.path.join(raw_data_folder_path, "s-and-p-500-companies.csv")
    stocks_prices_folder_path = os.path.join(raw_data_folder_path, "Stocks")

    if not Path(stocks_prices_folder_path).is_dir():
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
            path=f"{raw_data_folder_path}",
            unzip=True,
        )

    if not Path(s_and_p_list_path).is_file():
        urllib.request.urlretrieve(
            "https://datahub.io/core/s-and-p-500-companies/r/0.csv", s_and_p_list_path
        )

    return s_and_p_list_path, stocks_prices_folder_path


def get_stocks_symbols(s_and_p_list_path: str) -> List[str]:
    return list(pd.read_csv(s_and_p_list_path)["Symbol"].str.lower())


def load_stocks_details(stocks_symbols: List[str], stocks_prices_folder_path: str) -> pd.DataFrame:
    dfs = []
    for symbol in stocks_symbols:
        file_path = os.path.join(stocks_prices_folder_path, f"{symbol}.us.txt")
        if Path(file_path).is_file():
            df = pd.read_csv(file_path)
            df["symbol"] = symbol

            dfs.append(df)

    return pd.concat(dfs)


def data_preprocessing(s_and_p_list_path: str, stocks_prices_folder_path: str):
    stocks_symbols = get_stocks_symbols(s_and_p_list_path)
    stocks_df = load_stocks_details(stocks_symbols, stocks_prices_folder_path)



def main():
    s_and_p_list_path, stocks_prices_folder_path = data_fetching()
    data_preprocessing(s_and_p_list_path, stocks_prices_folder_path)


if __name__ == "__main__":
    main()
