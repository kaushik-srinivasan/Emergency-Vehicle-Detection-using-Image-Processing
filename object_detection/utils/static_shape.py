from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_dim_as_int(dim):
  """Utility to get v1 or v2 TensorShape dim as an int."""
  try:
    return dim.value
  except AttributeError:
    return dim


def get_batch_size(tensor_shape):
  """Returns batch size from the tensor shape."""
  tensor_shape.assert_has_rank(rank=4)
  return get_dim_as_int(tensor_shape[0])


def get_height(tensor_shape):
  """Returns height from the tensor shape."""
  tensor_shape.assert_has_rank(rank=4)
  return get_dim_as_int(tensor_shape[1])


def get_width(tensor_shape):
  """Returns width from the tensor shape."""
  tensor_shape.assert_has_rank(rank=4)
  return get_dim_as_int(tensor_shape[2])


def get_depth(tensor_shape):
  """Returns depth from the tensor shape."""
  tensor_shape.assert_has_rank(rank=4)
  return get_dim_as_int(tensor_shape[3])
