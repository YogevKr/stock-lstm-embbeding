import argparse
import pickle

import torch
import numpy as np

from main import StockNN


def inference(net: StockNN, last_window_data: np.array, inference_period: int):
    data = last_window_data.copy()
    net.eval()
    for _ in range(inference_period):
        seq = torch.from_numpy(data).float()
        with torch.no_grad():
            net.hidden = (
                torch.zeros(1, data.shape[0], net.hidden_layer_size),
                torch.zeros(1, data.shape[0], net.hidden_layer_size),
            )
            last_window_data = np.concatenate(data, net(seq).item())

    return data[last_window_data.shape[0] :, :]


def main(args):
    with open(args.trained_model_name, "rb") as f:
        net = pickle.load(f)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=float, default=30)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--trained_model_name", type=str, default="./res/1000Epochs_w_emb_net.pkl")

    args, _ = parser.parse_known_args()
    main(args)
