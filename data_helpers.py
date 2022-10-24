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

    y = tx.dot(w)

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
    column_ids = [14, 17, 24, 27]

    for c in column_ids:
        tx[:, c] = abs(tx[:, c])

    return tx

def standardize(tx):
    """
    Standardize the original data set
    """
    mean, std = np.mean(tx, axis=0), np.std(tx, axis=0)

    return (tx - mean) / std

def heavy_tail(x, column_ids):
    """
    Compute the log transformation for heavy-tailed features
    """

    for id in column_ids:
        x[:, id] = np.log1p(x[:, id])

    return x

def drop_columns(x, column_ids):
    """
    Delete the column representing 
    """

    return np.delete(x, column_ids, axis=1)

def pre_processing(x, idx):
    """
    Wrapper functions to prepare and preprocess the data
    """

    cols_to_drop = {
        0 : [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],
        1 : [4, 5, 6, 12, 22, 26, 27, 28],
        2 : [22],
        3 : [22],
    }
    cols_to_log = {
        0 : [0, 1, 2, 3, 8, 9, 13, 16, 19, 21],
        1 : [0, 1, 2, 3, 8, 9, 13, 16, 19, 21, 23, 29],
        2 : [0, 1, 2, 3, 8, 9, 13, 16, 19, 21, 23, 26, 29],
        3 : [0, 1, 2, 3, 8, 9, 13, 16, 19, 21, 23, 26, 29],
    }

    x = cleaning_data(x)

    x = abs_transform(x)

    x = heavy_tail(x, cols_to_log[idx])

    x = drop_columns(x, cols_to_drop[idx])

    x = delete_outliers(x)

    x = standardize(x)

    return x