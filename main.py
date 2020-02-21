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
    def __init__(self):
        super(StockNN, self).__init__()
        self.embeds = nn.Embedding(NUM_OF_STOCKS, EMBEDDING_SIZE)
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_SIZE + INPUT_SIZE,
            hidden_size=LSTM_HID_SIZE,
            num_layers=1,
        )
        self.fc = nn.Linear(LSTM_HID_SIZE, INPUT_SIZE)

    def forward(self, stock_idx, price):
        x = torch.cat(
            (self.embeds(torch.tensor(stock_idx)), torch.FloatTensor(price)), dim=1
        ).view(-1, 1, EMBEDDING_SIZE + INPUT_SIZE)

        out, hidden = self.lstm(x)

        out = out.contiguous().view(-1, LSTM_HID_SIZE)
        # out = out.view(-1, self.num_flat_features(out))
        out = self.fc(out)

        return out, hidden


def train_nn(optimizer, net, data, num_of_epochs=10, print_every=200):
    (x_train, y_train, x_test, y_test) = data
    stock_inx = 1  # TODO: Fix
    train_loss_tracking = []
    test_loss_tracking = []
    for epoch in range(num_of_epochs):
        train_loss = 0.0
        tot_train_loss = 0.0
        for i in range(len(x_train)):

            # get the inputs
            # inputs, labels = data
            inputs = x_train[i]
            outputs = y_train[i]

            # inputs = inputs.cuda()  # -- For GPU
            # labels = labels.cuda()  # -- For GPU

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # TODO: Pass matrix
            out = net(np.repeat(stock_inx, inputs.shape[0]), inputs)
            loss = criterion(out[0], outputs[-1])
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            tot_train_loss += loss.item()
            if ((i + 1) % print_every) == 0:
                print(
                    "[{:4d}, {:5d}] loss: {:.3f}".format(
                        epoch + 1, i + 1, train_loss / print_every
                    )
                )
                train_loss = 0.0
        train_loss_tracking.append(tot_train_loss / i)

        # test_loss = 0.0
        # tot_test_loss = 0.0
        # for i, data in enumerate(testloader, 0):
        #     # get the inputs
        #     inputs, labels = data
        #
        #     inputs = inputs.cuda()  # -- For GPU
        #     labels = labels.cuda()  # -- For GPU
        #
        #     # zero the parameter gradients
        #     optimizer.zero_grad()
        #
        #     # forward + backward + optimize
        #     outputs = net(inputs)
        #     loss = criterion(outputs, labels)
        #     test_loss += loss.item()
        # print('Test: [{:4d}] loss: {:.3f}'.format(epoch + 1, test_loss / len(testloader)))
        # test_loss_tracking.append(test_loss / i)

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
