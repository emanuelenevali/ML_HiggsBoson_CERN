#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)
    """

    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(tx, w):
    """
    Return prediction given the data and the weights
    """

    y = tx.dot(w)

    y[np.where(y <= 0)] = -1
    y[np.where(y > 0)] = 1

    return y


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def cleaning_data(tx, nan_val=-999):
    """
    Substitute with the median the NaN values (-999)
    """
    D = tx.shape[1]

    for i in range(D):
        median = np.median(tx[:, i][tx[:, i] != nan_val])
        tx[:, i] = np.where(tx[:, i] == nan_val, median, tx[:, i])

    return tx


def delete_outliers(tx, a=0.05):
    """
    Delete the tails of tx given the quantile a (a=5% by default)
    """
    D = tx.shape[1]

    for i in range(D):
        tx[:, i][tx[:, i] < np.quantile(tx[:, i], a)] = np.quantile(tx[:, i], a)
        tx[:, i][tx[:, i] > np.quantile(tx[:, i], 1 - a)] = np.quantile(tx[:, i], 1 - a)

    return tx


def get_mask(tx):
    """
    Create a mask for each distinct value of jet (0,1,2,3)
    """
    jet_column = 22

    return [
        tx[:, jet_column] == 0,
        tx[:, jet_column] == 1,
        tx[:, jet_column] == 2,
        tx[:, jet_column] == 3,
    ]


def abs_transform(tx):
    """
    Apply the absolute value to features symmetrical distributed around 0
    """
    column_ids = [14, 17, 24, 27]

    tx[:, column_ids] = abs(tx[:, column_ids])

    return tx


def standardize(tx, mean, std):
    """
    Standardize the original data set
    """
    return (tx - mean)[:, std != 0] / std[std != 0]


def heavy_tail(x, idx):
    """
    Compute the log transformation for heavy-tailed features
    """
    cols_to_log = {
        0: [0, 1, 2, 3, 8, 9, 13, 16, 19, 21],
        1: [0, 1, 2, 3, 8, 9, 13, 16, 19, 21, 23, 29],
        2: [0, 1, 2, 3, 8, 9, 13, 16, 19, 21, 23, 26, 29],
        3: [0, 1, 2, 3, 8, 9, 13, 16, 19, 21, 23, 26, 29],
    }

    x[:, cols_to_log[idx]] = np.log1p(x[:, cols_to_log[idx]])

    return x


def cos_angles(x):
    """
    Tranformation for angles features
    """
    column_ids = [15, 18, 20, 25, 28]

    x[:, column_ids] = np.cos(x[:, column_ids])

    return x


def drop_columns(x, idx=None, cols_to_drop=None):
    """
    Delete the column with just -999 for each subset
    """

    if cols_to_drop is None:
        cols_to_drop = {
            0: [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],
            1: [4, 5, 6, 12, 22, 26, 27, 28],
            2: [22],
            3: [22],
        }
        cols_to_drop = cols_to_drop[idx]

    else:
        if len(cols_to_drop) == 0:
            return x
        cols_to_drop = cols_to_drop[0]

    return np.delete(x, cols_to_drop, axis=1)


def pre_processing(x_tr, x_te, idx):
    """
    Wrapper functions to prepare and preprocess the data
    """

    x_tr = cleaning_data(x_tr)
    x_te = cleaning_data(x_te)

    x_tr = abs_transform(x_tr)
    x_te = abs_transform(x_te)

    x_tr = delete_outliers(x_tr)
    x_te = delete_outliers(x_te)

    x_tr = cos_angles(x_tr)
    x_te = cos_angles(x_te)

    x_tr = heavy_tail(x_tr, idx)
    x_te = heavy_tail(x_te, idx)

    x_tr = drop_columns(x_tr, idx=idx)
    x_te = drop_columns(x_te, idx=idx)

    mean, std = np.mean(x_tr, axis=0), np.std(x_tr, axis=0)

    x_tr = standardize(x_tr, mean, std)
    x_te = standardize(x_te, mean, std)

    return x_tr, x_te
