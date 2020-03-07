import argparse
import os
import pickle

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from main import (
    EmbeddingLstm,
    load_data,
    convert_unique_idx,
    device,
    split_data_to_windows,
    OneHotLstm)
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
        self.df = test_data
        self.symbol_idx_mapping = symbol_idx_mapping
        self.train_last_windows = train_last_windows
        self.scaled_test_data = None
        self.scalers = None

        self.process_test_set()

    def process_test_set(self):
        self.scaled_test_data, self.scalers = split_data_to_windows(self.df, 30)

    def __getitem__(self, idx):
        return self.scaled_test_data[idx]

    def __len__(self):
        return len(self.scaled_test_data)


def inference(
    net: EmbeddingLstm, stock_idx: torch.Tensor, last_window_data: torch.Tensor
):
    net.eval()
    with torch.no_grad():
        net.hidden_cell = (
            torch.zeros(1, last_window_data.shape[0], net.hidden_layer_size).to(device),
            torch.zeros(1, last_window_data.shape[0], net.hidden_layer_size).to(device),
        )
        return net(stock_idx, last_window_data)


def calculate_error(y, y_hat):
    y = np.array(y) + 1
    y_hat = np.array(y_hat) + 1
    error_ = np.sqrt(np.mean(np.square((y - y_hat) / y)))
    return error_


def calculate_test_set_error(
    net, window_size, train_data_df, test_data_df, symbol_idx_mapping, batch_size
):
    train_last_windows = (
        train_data_df.set_index("symbol")["Close"]
        .apply(lambda x: x[-window_size:])
        .to_dict()
    )

    test_set = TestDataset(
        train_data=train_data_df,
        test_data=test_data_df,
        symbol_idx_mapping=symbol_idx_mapping,
        train_last_windows=train_last_windows,
    )

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    results = defaultdict(lambda: defaultdict(list))
    for i, (idx, last_window, y) in tqdm(enumerate(loader)):
        r = inference(
            net=net, stock_idx=idx.to(device), last_window_data=last_window.to(device)
        )

        r_ = r.cpu().numpy()
        for j, idx_ in enumerate(idx):
            sym_index = idx_[0].cpu().numpy().tolist()
            results[sym_index]["y"].append(y[j])
            results[sym_index]["y_hat"].append(r_[j])

    for k in results.keys():
        results[k]["error"] = calculate_error(
            y=results[k]["y"], y_hat=results[k]["y_hat"]
        )

    return results


def main(args):
    state_dict = torch.load(args.artifacts_dir, map_location=device)

    if state_dict.get("model_type") == "EmbeddingLstm":
        net = EmbeddingLstm(452)
    elif state_dict.get("model_type") == "OneHotLstm":
        net = OneHotLstm(452)
    else:
        net = EmbeddingLstm(452)

    net.load_state_dict(state_dict)

    artifacts_path = os.path.join(args.artifacts_dir, "data.pickle")
    with open(artifacts_path, "rb") as f:
        train_artifacts = pickle.load(f)

    res = calculate_test_set_error(
        net.to(device),
        args.window_size,
        train_artifacts.get("train_data_df"),
        train_artifacts.get("test_data_df"),
        train_artifacts.get("symbol_idx_mapping"),
        batch_size=args.batch_size,
    )
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="./runs/Mar06_09-07-03_ip-172-31-39-84NET_EmbeddingLstm_EPOCHS50_LR_0.001_BATCH_1024/",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/17/model.pth",
    )

    parser.add_argument("--embedding", type=bool, default=True)

    args, _ = parser.parse_known_args()
    main(args)
