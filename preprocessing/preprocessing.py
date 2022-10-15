import numpy as np
def load_csv_data(PATH, sub_sample=False):
    """ Inputs: path: data_path, sub_sample: if True selects a sample of the data
        Outputs: y: labels, tx: features matrix and ids (event ids)"""
    y = np.genfromtxt(PATH, delimiter=",", skip_header=1, dtype=str, usecols=1)
    data = np.genfromtxt(PATH, delimiter=",", skip_header=1)
    tx = data[:, 2:]

    # strings labels to binary (-1,1)
    labels = np.ones(len(y))
    labels[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        tx= tx[::50]
        labels=labels[::50]

    return tx, labels


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x


def data_cleaning(tx):
    """
    Deletion of features with more than 70% missing values and imposition of the median in the remaining features
    """
    N, D = tx.shape
    missing_data = np.zeros(D)
    cols_to_delete = []
    for i in range(D):
        missing_data[i] = np.count_nonzero(tx[:,i]==-999)/N

        if missing_data[i]>0.7:
            cols_to_delete.append(i)

        elif missing_data[i]>0:
            tx_feature = tx[:,i]
            median = np.median(tx_feature[tx_feature != -999])
            tx[:,i] = np.where(tx[:,i]==-999, median, tx[:,i])

    tx[:,cols_to_delete]=0

    return tx