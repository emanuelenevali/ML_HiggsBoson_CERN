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


