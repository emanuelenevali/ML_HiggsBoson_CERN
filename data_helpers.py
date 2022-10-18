import csv
import numpy as np

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
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

def cleaning_data(tx):
    """ preprocessing data: delete columns with more than 50% missing values or substitute median otherwise"""
    N,D=tx.shape
    for i in range(D):
        median=np.median(tx[:,i][tx[:,i]!=-999])
        bad = np.count_nonzero(tx[:,i]==-999)
        if bad>=0.5*N:
            tx[:,i]=0
        tx[:,i]=np.where(tx[:,i]==-999,median,tx[:,i])
    return tx    
        
def predict_labels(tx, w):
    """Return prediction given the data and the weights"""
    y = np.dot(tx, w)
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
            
            
def delete_outliers(tx, a):
    """
    Delete the tails of tx given the quantile a
    """
    for i in range(tx.shape[1]):
        tx[:,i][tx[:,i]<np.quantile(tx[:,i],a)] = np.quantile(tx[:,i],a)
        tx[:,i][tx[:,i]>np.quantile(tx[:,i],1-a)] = np.quantile(tx[:,i],1-a)
    return tx


def standardize(tx, mean=None, std=None):
    """
    Standardize the original data set
    """
    mean = np.mean(tx)
    tx = tx - mean
    std = np.std(tx)
    tx = tx / std

    return tx

def pre_processing(x_tr, x_te, degree, gamma):
    """Process the training and test set
    
    Args:
        x: numpy array that contains the data points for training.
        x_te: matrix that contains the data points for testing.
        degree: degree of polynomial expansion
        gamma: percentile from which to start clamping outliers on both sides
    
    Returns:
        x_tr: training data.
        x_te: testing data.

    """

    return x_tr, x_te