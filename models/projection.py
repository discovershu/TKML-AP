import numpy as np
import six
from six.moves import xrange
# import tensorflow as tf
import torch

def clip_eta_numpy(eta, ord, eps):
    avoid_zero_div = 1e-6
    if ord==2:
        # avoid_zero_div must go inside sqrt to avoid a divide by zero
        # in the gradient through this operation
        norm = np.maximum(avoid_zero_div, np.linalg.norm(eta))
        # We must *clip* to within the norm ball, not *normalize* onto the
        # surface of the ball
        factor = np.minimum(1., eps/norm)
        eta = eta * factor
    return eta


def clip_eta(eta, ord, eps):
    # reduc_ind = list(xrange(1, len(eta.get_shape())))
    reduc_ind = list(xrange(1, len(eta.shape)))
    avoid_zero_div = 1e-12
    if ord==2:
        # avoid_zero_div must go inside sqrt to avoid a divide by zero
        # in the gradient through this operation
        # norm = torch.sqrt(torch.max(avoid_zero_div, torch.sum(torch.square(eta), reduc_ind, keepdims=True)))
        norm = torch.sqrt(torch.clamp(torch.sum(torch.square(eta), reduc_ind, keepdims=True), min=avoid_zero_div) )
        # We must *clip* to within the norm ball, not *normalize* onto the
        # surface of the ball
        # factor = torch.minimum(1., div(eps, norm))
        factor = torch.clamp(div(eps, norm), max=1.0)
        eta = eta * factor
    elif ord==np.inf:
        eta =torch.sign(eta)*torch.clamp(torch.abs(eta), max=eps)
    return eta

def div(a, b):
  """
  A wrapper around tf division that does more automatic casting of
  the input.
  """
  def divide(a, b):
    """Division"""
    return a / b
  return op_with_scalar_cast(a, b, divide)

def op_with_scalar_cast(a, b, f):
  """
  Builds the graph to compute f(a, b).
  If only one of the two arguments is a scalar and the operation would
  cause a type error without casting, casts the scalar to match the
  tensor.
  :param a: a tf-compatible array or scalar
  :param b: a tf-compatible array or scalar
  """

  try:
    return f(a, b)
  except (TypeError, ValueError):
    pass

  def is_scalar(x):
    """Return True if `x` is a scalar"""
    if hasattr(x, "get_shape"):
      shape = x.get_shape()
      return shape.ndims == 0
    if hasattr(x, "ndim"):
      return x.ndim == 0
    assert isinstance(x, (int, float))
    return True

  a_scalar = is_scalar(a)
  b_scalar = is_scalar(b)

  if a_scalar and b_scalar:
    raise TypeError("Trying to apply " + str(f) + " with mixed types")

  if a_scalar and not b_scalar:
    a = torch.cast(a, b.dtype)

  if b_scalar and not a_scalar:
    b = torch.cast(b, a.dtype)

  return f(a, b)
