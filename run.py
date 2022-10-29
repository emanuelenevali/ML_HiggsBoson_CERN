#!/usr/bin/python3
# -*- coding: utf-8 -*-

from implementations import *
from data_helpers import *
import warnings

warnings.simplefilter("ignore")

paths = {
    "train": "data/train.csv",
    "test": "data/test.csv",
    "submission": "data/sample-submission.csv",
}

N = 4

# For each of the 4 subsets, respectively best lambda and degree
hyper_params = [(1e-05, 7), (1e-06, 6), (1e-06, 6), (0.001, 6)]


def load_and_prepare_data():
    """
    - Load the data from the csv (both train and test)
    - Find the mask to split into 4 subsets
    - Preprocess each of the subset
    """

    y_tr, tx_tr, _ = load_csv_data(paths["train"], sub_sample=False)
    y_te, tx_te, ids_te = load_csv_data(paths["test"])

    len_test = len(y_te)

    y_tr = y_tr[:, np.newaxis]
    y_pred = np.zeros(len_test)

    mask_tr = get_mask(tx_tr)
    mask_te = get_mask(tx_te)

    x_tr_subsamples = []
    y_tr_subsamples = []

    x_te_subsamples = []

    for i in range(N):
        x_tr_subsamples.append(tx_tr[mask_tr[i]])
        y_tr_subsamples.append(y_tr[mask_tr[i]])
        x_te_subsamples.append(tx_te[mask_te[i]])

    for j in range(N):
        x_tr_subsamples[j], x_te_subsamples[j] = pre_processing(
            x_tr_subsamples[j], x_te_subsamples[j], j
        )

    return x_tr_subsamples, x_te_subsamples, y_tr_subsamples, y_pred, mask_te, ids_te


def train_model(txs, ys):
    """
    Trains the classifier model

    Args:
        txs: training data split into 4 subsets
        y: labels of training data split into 4 subsets
        params: lambda and degree of each subset

    Returns:
        ws: weights of each subsets.
    """

    ws = []

    for i in range(len(txs)):

        lambda_, degree = hyper_params[i]
        x_poly = build_poly(txs[i], degree)

        ws.append(ridge_regression(ys[i], x_poly, lambda_=lambda_)[0])

    return ws

def train_set_accuracy(ws, x_tr, y_tr):

    correct = np.array((0, ))
    for i in range(N):
        degree = hyper_params[i][1]
        tmp = y_tr[i] == predict_labels(build_poly(x_tr[i], degree), ws[i])
        correct = np.vstack((correct, tmp))

    acc = np.mean(correct)

    print(f'Accuracy on train set: {np.around(acc, 3)}')


def generate_predictions(txs_te, ws, mask_test, y_pred):
    """
    Generate the predictions and save ouput
    """

    for j in range(len(txs_te)):
        degree = hyper_params[j][1]
        y_pred[mask_test[j]] = [
            y[0] for y in predict_labels(build_poly(txs_te[j], degree), ws[j])
        ]


def main():
    """
    Main function: load and prepare the data, train the model with
    a ridge regression and generate in the csv the predictions
    """

    (
        x_tr_subsamples,
        x_te_subsamples,
        y_tr_subsamples,
        y_pred,
        mask_te,
        ids_te,
    ) = load_and_prepare_data()

    ws = train_model(x_tr_subsamples, y_tr_subsamples)

    train_set_accuracy(ws, x_tr_subsamples, y_tr_subsamples)

    generate_predictions(x_te_subsamples, ws, mask_te, y_pred)

    create_csv_submission(ids_te, y_pred, paths["submission"])


if __name__ == "__main__":
    main()
