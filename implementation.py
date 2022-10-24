#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from model_helpers import *

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
        
        # compute gradient
        gradient = compute_gradient(y,tx,w)
        # update w by gradient descent
        w = w - gamma * gradient
        # compute loss
        loss = compute_loss(y, tx, w)

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

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):

            # compute a stochastic gradient
            grad = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad

        # calculate loss
        loss = compute_loss(y, tx, w)
   
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

    return (w := np.linalg.solve(a,b)), compute_loss(y, tx, w)

def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar denoting the loss computed as MSE

    """
    n, d = tx.shape[0], tx.shape[1]

    aI = 2 * n * lambda_ * np.eye(d)
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    
    return (w := np.linalg.solve(a, b)), compute_loss(y, tx, w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implement logistic regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar denoting the loss

    """
    w = initial_w
    for _ in range(max_iters):
        loss = lr_calculate_loss(y,tx,w)
        g = lr_calculate_gradient(y,tx,w)
        w = w - gamma*g
 
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Implement regularized logistic linear.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: a scalar denoting the lambda used for regularization.
        initial_w: initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar denoting the loss

    """
    w = initial_w
    for _ in range(max_iters):
        loss = reg_lr_compute_loss(y,tx,w,lambda_)
        g = reg_lr_compute_gradient(y,tx,w,lambda_)
        w = w - gamma*g

    return w, loss