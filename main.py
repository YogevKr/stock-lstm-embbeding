import argparse
import json
import os
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingLstm(nn.Module):
    def __init__(
        self,
        num_of_stocks,
        embedding_dim=2,
        hidden_layer_size=100,
        input_size=1,
        output_size=1,
    ):
        super(EmbeddingLstm, self).__init__()
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


class OneHotLstm(nn.Module):
    def __init__(
        self, num_of_stocks, hidden_layer_size=100, input_size=1, output_size=1
    ):
        super(OneHotLstm, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_of_stocks = num_of_stocks

        self.lstm = nn.LSTM(
            input_size=num_of_stocks + input_size,
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

        # shape: [seq_len, batch_size, input_size]
        input_seq = torch.cat(
            (
                F.one_hot(stock_idx, self.num_of_stocks)
                .view(stock_idx.shape[1], -1, self.num_of_stocks)
                .float(),
                price.view(price.shape[1], -1, 1),
            ),
            dim=2,
        )

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.fc(lstm_out[-1, :, :])
        return predictions


def pickle_tracks(data: dict, path_dir: str):
    path = os.path.join(path_dir, "data.pickle")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def train(
    net: nn.Module,
    data_loader: DataLoader,
    num_of_epochs: int = 10,
    print_every_batches: int = 10,
    train_data_df=None,
    test_data_df=None,
    symbol_idx_mapping=None,
    window_size=30,
    learning_rate=0.001,
    evaluation_batch_size=1024,
    batch_size=1024,
):
    writer = SummaryWriter(
        comment=f"NET_{type(net).__name__}_EPOCHS{num_of_epochs}_LR_{learning_rate}_BATCH_{batch_size}"
    )
    writer.add_hparams(
        {
            "net:": type(net).__name__,
            "lr": learning_rate,
            "batch_size": batch_size,
            "epochs": num_of_epochs,
        },
        {},
    )

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_epoch_loss_tracking = []
    test_error_tracking = []

    for epoch in range(num_of_epochs):
        epoch_loss = 0.0

        for batch_idx, (symbols, inputs, labels) in enumerate(data_loader):

            # zero the parameter gradients
            net.train()
            optimizer.zero_grad()
            net.hidden_cell = (
                torch.zeros(1, inputs.shape[0], net.hidden_layer_size).to(device),
                torch.zeros(1, inputs.shape[0], net.hidden_layer_size).to(device),
            )

            # Forward
            y_pred = net(symbols.to(device), inputs.to(device))
            # Backward
            loss = criterion(y_pred, labels.view(-1, 1).to(device))
            loss.backward()
            # Back Prop
            optimizer.step()

            # print statistics
            writer.add_scalar(
                "batch_train_loss",
                loss.item(),
                global_step=(epoch + 1) * (batch_idx + 1),
            )
            epoch_loss += loss.item()
            if (batch_idx + 1) % print_every_batches == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Batch Loss: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * data_loader.batch_size,
                        len(data_loader.sampler),
                        100.0
                        * (batch_idx + 1)
                        * data_loader.batch_size
                        / len(data_loader.dataset),
                        loss.item(),
                    )
                )

        writer.add_scalar(
            "epochs_train_loss",
            epoch_loss / len(data_loader.dataset),
            global_step=(epoch + 1),
        )
        train_epoch_loss_tracking.append(epoch_loss / len(data_loader.dataset))

        try:
            writer.add_embedding(
                net.embeds.weight,
                metadata=list(symbol_idx_mapping.keys()),
                global_step=(epoch + 1),
            )
        except AttributeError:
            pass

        from inference import calculate_test_set_error

        res = calculate_test_set_error(
            net,
            window_size,
            train_data_df,
            test_data_df,
            symbol_idx_mapping,
            batch_size=evaluation_batch_size,
        )
        mean_test_error = np.mean([x["error"] for x in res.values()])
        test_error_tracking.append(mean_test_error)
        writer.add_scalar("mean_test_error", mean_test_error)
        print(test_error_tracking[-1])

    writer.add_hparams({"final_mean_test_error": np.mean(test_error_tracking)})

    save_model(net, writer.log_dir)
    pickle_tracks(
        dict(
            train_epoch_loss_tracking=train_epoch_loss_tracking,
            test_error_tracking=test_error_tracking,
        ),
        writer.log_dir
    )
    print("Finished Training")
    writer.close()
    return train_epoch_loss_tracking, test_error_tracking


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
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[np.ndarray]]:
    train_data = []
    scalers = []
    for symbol, prices, idx in train_data_df.itertuples(index=False):
        prices = list(map(np.float32, prices))

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

        x_scaled = x / x[:, :1]
        y_scaled = y / x[:, :1]

        assert x_scaled.shape[0] == y_scaled.shape[0]

        train_data.extend(
            [
                (np.repeat(idx, len(price)), price, label)
                for price, label in zip(x_scaled, y_scaled)
            ]
        )
        scalers.append(x[:, :1])

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


def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


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

    test_data_df["idx"] = test_data_df["symbol"].map(symbol_idx_mapping)
    test_data_df["idx"] = test_data_df["idx"].astype("int")

    train_data_df["Close"] = train_data_df["Close"].apply(lambda x: json.loads(x))
    test_data_df["Close"] = test_data_df["Close"].apply(lambda x: json.loads(x))

    train_data, _ = split_data_to_windows(train_data_df, args.window_size, step_size=1)
    dataset = TrainDataset(train_data)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.shuffle_samples
    )

    embedding_net = EmbeddingLstm(
        num_of_stocks=len(symbol_idx_mapping.keys()),
        embedding_dim=args.embedding_dim,
        hidden_layer_size=args.lstm_hidden_layer_size,
    ).to(device)

    one_hot_net = OneHotLstm(
        num_of_stocks=len(symbol_idx_mapping.keys()),
        hidden_layer_size=args.lstm_hidden_layer_size,
    ).to(device)

    embedding_train_loss_tracking = train(
        embedding_net,
        loader,
        num_of_epochs=args.num_of_epochs,
        print_every_batches=args.print_every_batches,
        train_data_df=train_data_df,
        test_data_df=test_data_df,
        symbol_idx_mapping=symbol_idx_mapping,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        evaluation_batch_size=args.evaluation_batch_size,
        batch_size=args.batch_size,
    )

    one_hot_train_loss_tracking = train(
        one_hot_net,
        loader,
        num_of_epochs=args.num_of_epochs,
        print_every_batches=args.print_every_batches,
        train_data_df=train_data_df,
        test_data_df=test_data_df,
        symbol_idx_mapping=symbol_idx_mapping,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
        evaluation_batch_size=args.evaluation_batch_size,
        batch_size=args.batch_size,
    )

    visualization(embedding_net, symbol_idx_mapping)

    print(embedding_train_loss_tracking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding_dim", type=int, default=4)
    parser.add_argument("--lstm_hidden_layer_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_of_epochs", type=int, default=15)
    parser.add_argument("--print_every_batches", type=int, default=15)
    parser.add_argument("--evaluation_batch_size", type=int, default=1024)
    parser.add_argument("--shuffle_samples", type=bool, default=False)

    args, _ = parser.parse_known_args()
    main(args)
