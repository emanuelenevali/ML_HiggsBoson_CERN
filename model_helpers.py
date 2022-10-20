import numpy as np
from data_helpers import *

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
        y:  numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w:  numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx, w)
    
    return compute_mse(e)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y:  numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w:  numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        gradient: an numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
        e:        scalar denotin the loss (MSE)
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
        y:  array that contains the correct values to be predicted.
        tx: matrix that contains the data points. 
        w:  array containing the linear parameters to test.
    
    Returns:
        loss: the loss for the given logistic linear parameters.

    """
    dot = tx@w
    return -np.mean(y*np.log(sigmoid(dot)) + (1-y)*np.log(1-sigmoid(dot)))

def lr_calculate_gradient(y, tx, w):
    """"Compute the logistic gradient.
    
    Args:
        y:  array that contains the correct values to be predicted.
        tx: atrix that contains the data points. 
        w:  array containing the linear parameters to test.
    
    Returns:
        gradient: the gradient for the given logistic parameters.

    """
    return (tx.T@(sigmoid(tx@w)-y)) / y.shape[0]

def reg_lr_compute_loss(y, tx, w, lambda_):
    """
    Computation of the regularized logistic loss (negative log likelihood)
    INPUTS: y = target, tx = sample matrix, w = weights vector
    OUTPUTS: evaluation of the loss
    """
    return lr_calculate_loss(y,tx,w) + lambda_*np.linalg.norm(w,2)*2


def reg_lr_compute_gradient(y, tx, w, lambda_):
    """
    Computation of the gradient of the regularized logistic loss (negative log likelihood)
    INPUTS: y = target, tx = sample matrix, w = weights vector
    OUTPUTS: evaluation of the gradient of the loss
    """
    return lr_calculate_gradient(y,tx,w) + 2*lambda_*w



def build_k_indices(num_row, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        num_row: a scalar
        k_fold:  K in K-fold, i.e. the fold num
        seed:    a scalar indicating the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """

    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def build_poly(tx, deg):
    """Polynomial aggregation (0-degree)"""
    N, D = tx.shape
    tx_poly = np.zeros(shape=(N,deg*D+1))
    tx_poly[:,0] = np.ones(N)
    for degree in range(1,deg+1):
        for i in range(D):
            tx_poly[:,D*(degree-1)+(i+1)] = np.power(tx[:,i],degree)
    return tx_poly

def compute_accuracy(y_prediction, y):
    """Given the predictions and the test data: computes accuracy"""
    corrects = 0
    for i, y_te in enumerate(y):
        if y_te == y_prediction[i]:
            corrects += 1
    return corrects / len(y)