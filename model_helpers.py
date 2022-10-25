import numpy as np
from data_helpers import *

def compute_mse(e):
    """
    Calculate the MSE

    Args:
        e: numpy array of shape=(N,  )

    Returns:
        the value of the loss (a scalar)
    """

    return np.mean(e**2) / 2

def compute_loss(y, tx, w):
    """
    Calculate the loss using MSE

    Args:
        y:  numpy array of shape=(N,  )
        tx: numpy array of shape=(N, D)
        w:  numpy array of shape=(D,  )

    Returns:
        the value of the loss (a scalar)
    """
    e = y - tx.dot(w)
    
    return compute_mse(e)

def compute_gradient(y, tx, w):
    """
    Computes the gradient at w
        
    Args:
        y:  numpy array of shape=(N,  )
        tx: numpy array of shape=(N, D)
        w:  numpy array of shape=(D,  )
        
    Returns:
        gradient: an numpy array of shape (D, ), containing the gradient of the loss at w.
        e:        scalar denotin the loss (MSE)
    """

    N = len(y)
    e = y - tx.dot(w)
    gradient = -1/N * tx.T.dot(e)

    return gradient

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset
    
    Args:
       y:           numpy array of shape=(N,  )
       tx:          numpy array of shape=(N, D)
       batch_size:  scalar denoting the size of a batch
       num_batches: scalar (number of batches of size batch_size)
       shuffle:     boolean (Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches)
       
    Returns:
        iterator which gives mini-batches of batch_size matching elements from y and tx
    """

    N = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(N))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, N)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(t):
    """
    Apply sigmoid function on t

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """

    return 1 / (1 + np.exp(-t))

def lr_calculate_loss(y, tx, w):
    """"
    Loss for logistic regression
    
    Args:
        y:  numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        w:  numpy array of shape=(D, 1)
    
    Returns:
        loss: scalar, the loss for the given logistic linear parameters
    """

    tx_w = tx.dot(w)

    return -np.mean(y*np.log(sigmoid(tx_w)) + (1-y)*np.log(1-sigmoid(tx_w)))

def lr_calculate_gradient(y, tx, w):
    """
    Compute the logistic gradient
    
    Args:
        y:  numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        w:  numpy array of shape=(D, 1)
    
    Returns:
        gradient: the gradient for the given logistic parameters

    """
    N = y.shape[0]

    return (tx.T.dot(sigmoid(tx.dot(w))-y)) / N

def reg_lr_compute_loss(y, tx, w, lambda_):
    """"
    Loss for regularized logistic regression.
    
    Args:
        y:        numpy array of shape=(N, 1)
        tx:       numpy array of shape=(N, D)
        w:        numpy array of shape=(D, 1)
        lambda_ : scalar
    
    Returns:
        loss: scalar, the loss for the given logistic linear parameters
    """

    reg_term = lambda_ * np.linalg.norm(w,2)**2

    return lr_calculate_loss(y,tx,w) + reg_term

def reg_lr_compute_gradient(y, tx, w, lambda_):
    """"
    Compute the regularized logistic gradient.
    
    Args:
        y:        numpy array of shape=(N, 1)
        tx:       numpy array of shape=(N, D)
        w:        numpy array of shape=(D, 1)
        lambda_ : scalar
    
    Returns:
        gradient: the gradient for the given logistic parameters

    """

    reg_term = 2 * lambda_ * w

    return lr_calculate_gradient(y,tx,w) + reg_term

def build_k_indices(num_row, k_fold, seed):
    """
    Build k indices for k-fold
    
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
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree
    
    Args:
        tx: numpy array of shape (N,D), N is the number of samples, D is the number of features
        deg: integer
        
    Returns:
        tx_poly: numpy array of shape (N,D*d+1)

    """
    N, D = tx.shape

    tx_poly = np.zeros(shape=(N,deg*D+1))
    tx_poly[:,0] = np.ones(N)

    for degree in range(1,deg+1):
        for i in range(D):
            tx_poly[:,D*(degree-1)+(i+1)] = np.power(tx[:,i],degree)

    return tx_poly