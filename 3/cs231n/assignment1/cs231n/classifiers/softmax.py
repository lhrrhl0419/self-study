import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  dxw = np.zeros_like(scores)
  for i in range(scores.shape[0]):
    sum = 0
    for j in range(scores.shape[1]):
      sum += np.e ** scores[i, j]
    for j in range(scores.shape[1]):
      dxw[i, j] = np.e ** scores[i, j] / sum
    dxw[i, y[i]] -= 1.0
    loss -= np.log(np.e ** scores[i, y[i]] / sum)
  for i in range(W.shape[0]):
    for j in range(W.shape[1]):
      dW[i, j] = np.dot(dxw[:, j], X[:, i])
  dW /= scores.shape[0]
  loss /= scores.shape[0]
  dW += reg * 2 * W
  loss += reg * np.sum(W ** 2)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  escores = np.e ** scores
  esums = np.sum(escores, axis = -1)
  loss -= np.sum(np.log(escores[list(range(X.shape[0])), y] / esums))
  dxw = escores / np.reshape(esums, (esums.shape[0], 1))
  dxw[list(range(X.shape[0])), y] -= 1.0
  dW = np.dot(X.swapaxes(0, 1), dxw)
  loss /= X.shape[0]
  dW /= X.shape[0]
  loss += reg * np.sum(W ** 2)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

