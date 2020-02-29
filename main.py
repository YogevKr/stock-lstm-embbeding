import argparse
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

torch.manual_seed(1)


class StockNN(nn.Module):
    def __init__(
        self,
        num_of_stocks,
        input_size=1,
        embedding_dim=16,
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
        predictions = self.fc(lstm_out[-1, :, :])
        return predictions


def train(
    net: StockNN,
    data_loader: DataLoader,
    num_of_epochs: int = 10,
    print_every: int = 200,
    train_data_df=None,
    test_data_df=None,
    symbol_idx_mapping=None,
    window_size=None
):

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_loss_tracking = []
    test_error_tracking = []

    for epoch in range(num_of_epochs):
        train_loss = 0.0
        tot_train_loss = 0.0

        for batch_idx, (symbols, inputs, labels) in enumerate(data_loader):

            # zero the parameter gradients
            net.train()
            optimizer.zero_grad()
            net.hidden_cell = (
                torch.zeros(1, inputs.shape[0], net.hidden_layer_size),
                torch.zeros(1, inputs.shape[0], net.hidden_layer_size),
            )

            # Forward
            y_pred = net(symbols, inputs)
            # Backward
            single_loss = criterion(y_pred, labels.view(-1, 1))
            single_loss.backward()
            # Back Prop
            optimizer.step()

            # print statistics
            train_loss += single_loss.item()
            tot_train_loss += single_loss.item()
            if (batch_idx + 1) % print_every == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Batch Loss: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * data_loader.batch_size,
                        len(data_loader.sampler),
                        100.0
                        * (batch_idx + 1)
                        * data_loader.batch_size
                        / len(data_loader.dataset),
                        train_loss / print_every,
                    )
                )
                train_loss = 0.0
        train_loss_tracking.append(tot_train_loss / batch_idx)

        from inference import calculate_test_set_error
        res = calculate_test_set_error(net, window_size, train_data_df, test_data_df, symbol_idx_mapping)
        test_error_tracking.append(np.mean([x["error"] for x in res.values()]))

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
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[MinMaxScaler]]:
    train_data = []
    scalers = []
    for symbol, prices, idx in train_data_df.itertuples(index=False):
        prices = list(map(np.float32, json.loads(prices)))

        # Split Data To Windows with step_size step
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
        ).reshape(-1, 1)

        # Normalize Data
        scaler = MinMaxScaler((-1, 1))
        scaler.fit(np.concatenate((x, y), axis=1).T)
        x_scaled = scaler.transform(x.T).T * 10
        y_scaled = scaler.transform(y.T).T * 10

        assert x_scaled.shape[0] == y_scaled.shape[0]

        train_data.extend(
            [
                (np.repeat(idx, len(price)), price, label)
                for price, label in zip(x_scaled, y_scaled)
            ]
        )
        scalers.append(scaler)

    return train_data, scalers


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df["idx"] = df[column_name].map(column_dict)
    df["idx"] = df["idx"].astype("int")
    assert df["idx"].min() == 0
    assert df["idx"].max() == len(column_dict) - 1
    return df, column_dict


def visualization(net, symbol_idx_mapping):
    from sklearn.manifold import TSNE

    labels = list(symbol_idx_mapping.keys())
    tokens = [net.embeds.weight[i].tolist() for i in range(len(labels))]
    tsne_model = TSNE(
        perplexity=40, n_components=2, init="pca", n_iter=2500, random_state=23
    )

    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(
            labels[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    plt.show()


def compare_two_stocks(a_prices, b_prices, a_name, b_name):
    plt.plot(a_prices, label=a_name)
    plt.plot(b_prices, label=b_name)
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.title(f"Compare between {a_name} and {b_name}")
    plt.grid()
    plt.legend(loc="best")


def main(args):
    train_data_df, test_data_df = load_data()
    train_data_df, symbol_idx_mapping = convert_unique_idx(train_data_df, "symbol")

    train_data, _ = split_data_to_windows(
        train_data_df, args.window_size, step_size=args.window_size
    )
    dataset = TrainDataset(train_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    train_data_df["Close"] = train_data_df["Close"].apply(lambda x: json.loads(x))
    test_data_df["Close"] = test_data_df["Close"].apply(lambda x: json.loads(x))

    net = StockNN(num_of_stocks=len(symbol_idx_mapping.keys()))
    train_loss_tracking = train(
        net,
        loader,
        num_of_epochs=10,
        print_every=10,
        train_data_df=train_data_df,
        test_data_df=test_data_df,
        symbol_idx_mapping=symbol_idx_mapping,
        window_size=args.window_size
    )

    visualization(net, symbol_idx_mapping)

    print(train_loss_tracking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=150)

    args, _ = parser.parse_known_args()
    main(args)
