"""Some helper functions for project 1."""
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


def compute_mse(e):
    """Calculate the MSE

    Args:
        e: numpy array of shape=(N, )

    Returns:
        the value of the loss (a scalar)
    """
    return np.mean(e**2) / 2

def compute_loss(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx, w)
    
    return compute_mse(e)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        gradient: an numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
        e: scalar denotin the loss (MSE)
    """
    e = y - np.dot(tx, w)
    gradient = -1/len(e) * np.dot(tx.T,e)
    return gradient

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
     """
     Generate a minibatch iterator for a dataset.
     Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
     Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
     Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
     Example of use :
     for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
         <DO-SOMETHING>
     """
     data_size = len(y)

     if shuffle:
         shuffle_indices = np.random.permutation(np.arange(data_size))
         shuffled_y = y[shuffle_indices]
         shuffled_tx = tx[shuffle_indices]
     else:
         shuffled_y = y
         shuffled_tx = tx
     for batch_num in range(num_batches):
         start_index = batch_num * batch_size
         end_index = min((batch_num + 1) * batch_size, data_size)
         if start_index != end_index:
             yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def sigmoid(t):
    """Applies sigmoid function on t.
    
    Args:
        t: element onto which the sigmoid needs to be applied.
    
    Returns:
        s(t): element to which the sigmoid function has been applied.
    """

    return 1 / (1 + np.exp(-t))

def lr_calculate_loss(y, tx, w):
    """"Loss for logistic regression.
    
    Args:
        y: array that contains the correct values to be predicted.
        tx: matrix that contains the data points. 
        w: array containing the linear parameters to test.
    
    Returns:
        loss: the loss for the given logistic linear parameters.

    """
    m = tx.dot(w)
    return np.sum(np.log(1 + np.exp(m)) - y*m)

def lr_calculate_gradient(y, tx, w):
    """"Compute the logistic gradient.
    
    Args:
        y: array that contains the correct values to be predicted.
        tx: atrix that contains the data points. 
        w: array containing the linear parameters to test.
    
    Returns:
        gradient: the gradient for the given logistic parameters.

    """
    return tx.T.dot((sigmoid(tx.dot(w)) - y))

def lr_gradient_descent_step(y, tx, w, gamma, lambda_):
    """Computes one step of gradient descent for the logistic regression.
    
    Args:
        y: array that contains the correct values to be predicted.
        tx: matrix that contains the data points. 
        w: array containing the linear parameters to test.
        gamma: the stepsize.
        lambda_: the lambda used for regularization. Default behavior is without regularization.
    
    Returns:
        w: the linear parameters.
        loss: the loss given w as parameters.

    """
    loss = lr_calculate_loss(y, tx, w) + lambda_/2 * np.power(np.linalg.norm(w), 2)
    gradient = lr_calculate_gradient(y, tx, w) + lambda_ * w
    w -= gamma * gradient
    return loss, w

def predict_labels(tx, w):
    """Return prediction given the data and the weights"""
    y = np.dot(tx, w)
    y[np.where(y <= 0)] = -1
    y[np.where(y > 0)] = 1
    return y