#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

"""
----- Private functions used as helpers -----
"""

def __compute_mse(e):
    """Calculate the MSE

    Args:
        e: numpy array of shape=(N, )

    Returns:
        the value of the loss (a scalar)
    """
    return 1/2*np.mean(e**2)

def __compute_loss(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx, w)
    
    return __compute_mse(e)


def __compute_gradient(y, tx, w):
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
    return gradient, e

def __batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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

"""
----- Start of the public functions to deliver -----
"""

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using the gradient descent and the mean squared error as loss function.
       It returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar denoting the loss computed as MSE

    """

    w = initial_w
    
    for _ in range(max_iters):
        
        # compute loss, gradient
        gradient, err = __compute_gradient(y,tx,w)
        loss = __compute_mse(err)
        
        # update w by gradient descent
        w = w - gamma * gradient

    # return last weights and loss
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
                
        Args:
            y: numpy array of shape=(N, )
            tx: numpy array of shape=(N,2)
            initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
            max_iters: a scalar denoting the total number of iterations of SGD
            gamma: a scalar denoting the stepsize
            
        Returns:
            w: optimal weights, numpy array of shape(D,), D is the number of features.
            loss: scalar denoting the loss computed as MSE
    """
    
    # As specified in the requirements we use the standard mini-batch-size of 1
    batch_size = 1
    w = initial_w

    for _ in range(max_iters):

        for y_batch, tx_batch in __batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient
            grad, _ = __compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = __compute_loss(y, tx, w)
   
    # return last weights and loss
    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns optimal weights and mse.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar denoting the loss computed as MSE

    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return (w := np.linalg.solve(a,b)), __compute_loss(y, tx, w)

def ridge_regression(y, tx, _lambda):
    pass

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass

def reg_logistic_regression(y, tx, _lambda , initial_w, max_iters, gamma):
    pass