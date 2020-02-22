import argparse
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)


class StockNN(nn.Module):
    def __init__(
        self,
        num_of_stocks,
        input_size=1,
        embedding_dim=100,
        hidden_layer_size=100,
        output_size=1,
    ):
        super(StockNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.embeds = nn.Embedding(num_of_stocks, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + input_size,
            hidden_size=hidden_layer_size,
            num_layers=1,
        )
        self.fc = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, stock_idx, price):
        assert stock_idx.shape == price.shape
        input_seq = torch.cat(
            (self.embeds(stock_idx.T), price.view(price.shape[1], -1, 1)), dim=2
        )

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.fc(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train(net, data_loader, num_of_epochs=10, print_every=200):

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_loss_tracking = []
    test_loss_tracking = []
    for epoch in range(num_of_epochs):
        train_loss = 0.0
        tot_train_loss = 0.0

        for batch_idx, (symbols, inputs, labels) in enumerate(data_loader):

            # inputs = inputs.cuda()  # -- For GPU
            # labels = labels.cuda()  # -- For GPU

            # zero the parameter gradients
            optimizer.zero_grad()
            net.hidden_cell = (
                torch.zeros(1, 1, net.hidden_layer_size),
                torch.zeros(1, 1, net.hidden_layer_size),
            )

            # forward + backward + optimize
            # TODO: Pass matrix
            y_pred = net(symbols, inputs)
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()

            # print statistics
            train_loss += single_loss.item()
            tot_train_loss += single_loss.item()
            if ((batch_idx + 1) % print_every) == 0:
                print(
                    "[{:4d}, {:5d}] loss: {:.8f}".format(
                        epoch + 1, batch_idx + 1, train_loss / print_every
                    )
                )
                train_loss = 0.0
        # train_loss_tracking.append(tot_train_loss / batch_idx)

    print("Finished Training")
    return train_loss_tracking


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv("./data/processed/train_data.tsv", sep="\t")
    test_df = pd.read_csv("./data/processed/test_data.tsv", sep="\t")

    return train_df, test_df


class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def split_data_to_windows(
    train_data_df: pd.DataFrame, window_size: int, step_size: int = 1
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_data = []
    for symbol, prices, idx in train_data_df.itertuples(index=False):
        prices = list(map(np.float32, json.loads(prices)))
        x = np.array(
            [
                prices[i : i + window_size]
                for i in range(0, len(prices) - window_size, step_size)
            ]
        )
        y = np.array(
            [
                prices[i + window_size]
                for i in range(0, len(prices) - window_size, step_size)
            ]
        )

        assert x.shape[0] == y.shape[0]
        train_data.extend(
            [(np.repeat(idx, len(price)), price, label) for price, label in zip(x, y)]
        )

    return train_data


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df["idx"] = df[column_name].map(column_dict)
    df["idx"] = df["idx"].astype("int")
    assert df["idx"].min() == 0
    assert df["idx"].max() == len(column_dict) - 1
    return df, column_dict


def main(args):
    train_data_df, test_data_df = load_data()
    train_data_df, symbol_idx_mapping = convert_unique_idx(train_data_df, "symbol")

    train_data = split_data_to_windows(train_data_df, args.window_size)
    dataset = TrainDataset(train_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    net = StockNN(num_of_stocks=len(symbol_idx_mapping.keys()))
    train(net, loader, num_of_epochs=10, print_every=10)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=1)

    args, _ = parser.parse_known_args()
    main(args)
