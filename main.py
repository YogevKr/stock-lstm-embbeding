import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

torch.manual_seed(1)


NUM_OF_STOCKS = 100  # TODO: Total 505
EMBEDDING_SIZE = 8
LSTM_HID_SIZE = 128
INPUT_SIZE = 1
BATCH_SIZE = 1

# TODO: Dict that translates stock name to index


class StockNN(nn.Module):
    def __init__(
        self,
        num_of_stocks=NUM_OF_STOCKS,
        input_size=1,
        embedding_dim=EMBEDDING_SIZE,
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
        assert len(stock_idx) == len(price)
        input_seq = torch.cat(
            (self.embeds(torch.tensor(stock_idx)), torch.FloatTensor(price)), dim=1
        ).view(len(stock_idx), 1, -1)

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.fc(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_nn(optimizer, net, data, num_of_epochs=10, print_every=200):
    (x_train, y_train, x_test, y_test) = map(torch.FloatTensor, data)
    stock_inx = 1  # TODO: Fixd
    train_loss_tracking = []
    test_loss_tracking = []
    for epoch in range(num_of_epochs):
        train_loss = 0.0
        tot_train_loss = 0.0
        for i in range(len(x_train)):

            # get the inputs
            # inputs, labels = data
            inputs = x_train[i]
            labels = y_train[i]

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
            y_pred = net(np.repeat(stock_inx, inputs.shape[0]), inputs)
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()

            # print statistics
            train_loss += single_loss.item()
            tot_train_loss += single_loss.item()
            if ((i + 1) % print_every) == 0:
                print(
                    "[{:4d}, {:5d}] loss: {:.8f}".format(
                        epoch + 1, i + 1, train_loss / print_every
                    )
                )
                train_loss = 0.0
        train_loss_tracking.append(tot_train_loss / i)

    print("Finished Training")
    return train_loss_tracking


if __name__ == "__main__":
    with open("./data/data.pickle", "rb") as f:
        data = pickle.load(f, encoding="latin1")
    net = StockNN()

    NUM_OF_EPOCHS = 10
    PRINT_EVERY = 10
    LR = 0.001

    optimizer = optim.Adam(net.parameters(), lr=LR)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    train_nn(optimizer, net, data, num_of_epochs=10, print_every=10)

    # (x_train, y_train, x_test, y_test) = data
    # stock_idx = 1
    #
    # out, hidden = net([stock_idx for _ in range(len(x_train[0]))], x_train[0])

    print("Done")
