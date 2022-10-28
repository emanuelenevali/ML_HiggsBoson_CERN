#!/usr/bin/python3
# -*- coding: utf-8 -*-

from implementation import *
from data_helpers import *

paths = { 
            'train' : 'data/train.csv',
            'test' : 'data/test.csv',
            'submission' : 'data/sample-submission.csv' 
        }

N = 4

# For each of the 4 subsets, respectively best lambda and degree
hyper_params = [(1e-05, 9),
                (0.0001, 9),
                (1e-05, 9),
                (0.001, 6)]

def load_and_prepare_data():
    """
    - Load the data from the csv (both train and test)
    - Find the mask to split into 4 subsets
    - Preprocess each of the subset
    """

    y_tr, tx_tr, _ = load_csv_data(paths['train'], sub_sample=False)
    y_te, tx_te, ids_te = load_csv_data(paths['test'])

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
        x_tr_subsamples[j], x_te_subsamples[j] = pre_processing(x_tr_subsamples[j], x_te_subsamples[j], j)

    return x_tr_subsamples, x_te_subsamples, y_tr_subsamples, y_pred, mask_te, ids_te


def train_model(txs, ys, params):
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
        
        lambda_, degree = params[i]
        x_poly = build_poly(txs[i], degree)
        
        ws.append(ridge_regression(ys[i], x_poly, lambda_=lambda_)[0])
        
    return ws

def generate_predictions(txs_te, ws, mask_test, y_pred, params):
    """
    Generate the predictions and save ouput
    """
    
    for j in range(len(txs_te)):
        degree = params[j][1]
        y_pred[mask_test[j]] = [y[0] for y in predict_labels(build_poly(txs_te[j],degree), ws[j])]
            
    
def main():
    """
    Main function: load and prepare the data, train the model with 
    a ridge regression and generate in the csv the predictions
    """

    x_tr_subsamples, x_te_subsamples, y_tr_subsamples, y_pred, mask_te, ids_te = load_and_prepare_data()
    
    ws = train_model(x_tr_subsamples, y_tr_subsamples, hyper_params)

    generate_predictions(x_te_subsamples, ws, mask_te, y_pred, hyper_params)

    create_csv_submission(ids_te, y_pred, paths['submission'])


if __name__ == '__main__':
    main()
    