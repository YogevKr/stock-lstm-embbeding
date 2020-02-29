import argparse
import pickle

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from main import StockNN, load_data, convert_unique_idx
import json
from collections import defaultdict


class TestDataset(Dataset):
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        symbol_idx_mapping: dict,
        train_last_windows: dict,
    ):
        self.train_data = train_data
        self.test_data = test_data.set_index("symbol")
        self.symbol_idx_mapping = symbol_idx_mapping
        self.train_last_windows = train_last_windows

    def __getitem__(self, idx):
        symbol = self.test_data.index[idx]
        last_window = self.train_last_windows.get(symbol)
        return (
            symbol,  # Symbol
            np.repeat(self.symbol_idx_mapping.get(symbol), len(last_window)),  # idx
            last_window,  # Last window prices
            self.test_data.loc[symbol]["Close"],  # y
        )

    def __len__(self):
        return len(self.test_data.index)


def inference(
    net: StockNN, stock_idx: torch.Tensor, last_window_data: list, inference_period: int
):
    data = list(map(lambda x: float(x.numpy()), last_window_data.copy()))
    net.eval()
    for _ in range(inference_period):
        window = np.array(data[-len(last_window_data) :]).reshape(-1, 1)

        # Normalize Data
        scaler = MinMaxScaler((-1, 1))
        try:
            scaler.fit(window)
        except ValueError:
            pass
        window_scaled = scaler.transform(window) * 10
        seq = torch.FloatTensor(window_scaled).view(1, -1)
        with torch.no_grad():
            net.hidden_cell = (
                torch.zeros(1, 1, net.hidden_layer_size),
                torch.zeros(1, 1, net.hidden_layer_size),
            )
            data.append(float(scaler.inverse_transform(np.array([net(stock_idx, seq).item()]).reshape(1,-1))))

    return data[len(last_window_data) :]


def main(args):
    with open(args.trained_model_name, "rb") as f:
        net: StockNN = pickle.load(f)

    train_data_df, test_data_df = load_data()
    train_data_df, symbol_idx_mapping = convert_unique_idx(train_data_df, "symbol")

    train_data_df["Close"] = train_data_df["Close"].apply(lambda x: json.loads(x))
    test_data_df["Close"] = test_data_df["Close"].apply(lambda x: json.loads(x))

    train_last_windows = (
        train_data_df.set_index("symbol")["Close"]
        .apply(lambda x: x[-args.window_size :])
        .to_dict()
    )

    test_set = TestDataset(
        train_data=train_data_df,
        test_data=test_data_df,
        symbol_idx_mapping=symbol_idx_mapping,
        train_last_windows=train_last_windows,
    )

    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    results = defaultdict(lambda: defaultdict())
    for i, (symbol, idx, train_last_window, y) in tqdm(enumerate(loader)):
        results[symbol]["y"] = y
        results[symbol]["y_hat"] = inference(
            net=net,
            stock_idx=idx,
            last_window_data=train_last_window,
            inference_period=len(y),
        )

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument(
        "--trained_model_name", type=str, default="./res/1000Epochs_w_emb_net.pkl"
    )

    args, _ = parser.parse_known_args()
    main(args)
