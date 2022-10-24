import csv
import numpy as np
from pyparsing import col

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
    yb[np.where(y == "b")] = 0

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

    y = np.dot(tx, w)

    y[np.where(y <= 0.5)] = 0
    y[np.where(y > 0.5)] = 1

    return y

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    y_pred[np.where(y_pred == 0)] = -1
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
            
def delete_outliers(tx, a=.1):
    """
    Delete the tails of tx given the quantile a (a=10% by default)
    """

    for i in range(tx.shape[1]):
        tx[:, i][tx[:, i] < np.quantile(tx[:, i], a)] = np.quantile(tx[:, i], a)
        tx[:, i][tx[:, i] > np.quantile(tx[:, i], 1-a)] = np.quantile(tx[:, i], 1-a)

    return tx

def get_mask(tx):
    """
    Create a mask for each distinct value of jet (0,1,2,3)
    """
    jet_column = 22

    return [tx[:, jet_column] == 0, tx[:, jet_column] == 1, \
            tx[:, jet_column] == 2, tx[:, jet_column] == 3]

def abs_transform(tx):
    """
    Apply the absolute value to features symmetrical distributed around 0
    """
    column_ids = [14, 17, 23, 26]

    for c in column_ids:
        tx[:, c] = abs(tx[:, c])

    return tx

def standardize(tx, mean=None, std=None):
    """
    Standardize the original data set
    """
    mean, std = np.mean(tx), np.std(tx)

    return (tx - mean) / std

def heavy_tail(x):
    """
    Compute the log transformation for heavy-tailed features
    """
    column_ids = [0, 1, 2, 9, 13, 16, 19, 21, 22, 25]

    x_log1p = np.log1p(x[:, column_ids])

    # delete old columns
    np.delete(x_log1p, column_ids, axis=1)

    return np.hstack((x, x_log1p))

def del_jet_col(x):
    """
    Delete the column representing 
    """
    jet_col = 22

    return np.delete(x, jet_col, axis=1)

def pre_processing(x):
    """
    Wrapper functions to prepare and preprocess the data
    """

    x = cleaning_data(x)
    x = del_jet_col(x)
    x = heavy_tail(x)
    x = abs_transform(x)
    x = delete_outliers(x)
    x = standardize(x)

    return x