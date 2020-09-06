from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf

from object_detection.utils import static_shape


get_dim_as_int = static_shape.get_dim_as_int


def _is_tensor(t):
  """Returns a boolean indicating whether the input is a tensor"""
  return isinstance(t, (tf.Tensor, tf.SparseTensor, tf.Variable))


def _set_dim_0(t, d0):
  """Sets the 0-th dimension of the input tensor"""
  t_shape = t.get_shape().as_list()
  t_shape[0] = d0
  t.set_shape(t_shape)
  return t


def pad_tensor(t, length):
  """Pads the input tensor with 0s along the first dimension up to the length"""
  t_rank = tf.rank(t)
  t_shape = tf.shape(t)
  t_d0 = t_shape[0]
  pad_d0 = tf.expand_dims(length - t_d0, 0)
  pad_shape = tf.cond(
      tf.greater(t_rank, 1), lambda: tf.concat([pad_d0, t_shape[1:]], 0),
      lambda: tf.expand_dims(length - t_d0, 0))
  padded_t = tf.concat([t, tf.zeros(pad_shape, dtype=t.dtype)], 0)
  if not _is_tensor(length):
    padded_t = _set_dim_0(padded_t, length)
  return padded_t


def clip_tensor(t, length):
  """Clips the input tensor along the first dimension up to the length"""
  clipped_t = tf.gather(t, tf.range(length))
  if not _is_tensor(length):
    clipped_t = _set_dim_0(clipped_t, length)
  return clipped_t


def pad_or_clip_tensor(t, length):
  """Pad or clip the input tensor along the first dimension"""
  return pad_or_clip_nd(t, [length] + t.shape.as_list()[1:])


def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape"""
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor,
      begin=tf.zeros(len(clip_size), dtype=tf.int32),
      size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [
          tf.zeros(len(trailing_paddings), dtype=tf.int32),
          trailing_paddings
      ],
      axis=1)
  padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor


def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions"""
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
  """Runs map_fn as a (static) for loop when possible"""
  if isinstance(elems, list):
    for elem in elems:
      if not isinstance(elem, tf.Tensor):
        raise ValueError('`elems` must be a Tensor or list of Tensors.')

    elem_shapes = [elem.shape.as_list() for elem in elems]
    # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
    # to all be the same size along the batch dimension.
    for elem_shape in elem_shapes:
      if (not elem_shape or not elem_shape[0]
          or elem_shape[0] != elem_shapes[0][0]):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
    outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  else:
    if not isinstance(elems, tf.Tensor):
      raise ValueError('`elems` must be a Tensor or list of Tensors.')
    elems_shape = elems.shape.as_list()
    if not elems_shape or not elems_shape[0]:
      return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    outputs = [fn(arg) for arg in tf.unstack(elems)]
  # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
  if all([isinstance(output, tf.Tensor) for output in outputs]):
    return tf.stack(outputs)
  else:
    if all([isinstance(output, list) for output in outputs]):
      if all([all(
          [isinstance(entry, tf.Tensor) for entry in output_list])
              for output_list in outputs]):
        return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  raise ValueError('`fn` should return a Tensor or a list of Tensors.')


def check_min_image_dim(min_dim, image_tensor):
  """Checks that the image width/height are greater than some number"""
  image_shape = image_tensor.get_shape()
  image_height = static_shape.get_height(image_shape)
  image_width = static_shape.get_width(image_shape)
  if image_height is None or image_width is None:
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
                       tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
        ['image size must be >= {} in both height and width.'.format(min_dim)])
    with tf.control_dependencies([shape_assert]):
      return tf.identity(image_tensor)

  if image_height < min_dim or image_width < min_dim:
    raise ValueError(
        'image size must be >= %d in both height and width; image dim = %d,%d' %
        (min_dim, image_height, image_width))

  return image_tensor


def assert_shape_equal(shape_a, shape_b):
  """Asserts that shape_a and shape_b are equ"""
  if (all(isinstance(dim, int) for dim in shape_a) and
      all(isinstance(dim, int) for dim in shape_b)):
    if shape_a != shape_b:
      raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a, shape_b)


def assert_shape_equal_along_first_dimension(shape_a, shape_b):
  """Asserts that shape_a and shape_b are the same along the 0th-dimension"""
  if isinstance(shape_a[0], int) and isinstance(shape_b[0], int):
    if shape_a[0] != shape_b[0]:
      raise ValueError('Unequal first dimension {}, {}'.format(
          shape_a[0], shape_b[0]))
    else: return tf.no_op()
  else:
    return tf.assert_equal(shape_a[0], shape_b[0])


def assert_box_normalized(boxes, maximum_normalized_coordinate=1.1):
  """Asserts the input box tensor is normalized."""
  box_minimum = tf.reduce_min(boxes)
  box_maximum = tf.reduce_max(boxes)
  return tf.Assert(
      tf.logical_and(
          tf.less_equal(box_maximum, maximum_normalized_coordinate),
          tf.greater_equal(box_minimum, 0)),
      [boxes])


def flatten_dimensions(inputs, first, last):
  """Flattens `K-d` tensor along [first, last) dimensions"""
  if first >= inputs.shape.ndims or last > inputs.shape.ndims:
    raise ValueError('`first` and `last` must be less than inputs.shape.ndims. '
                     'found {} and {} respectively while ndims is {}'.format(
                         first, last, inputs.shape.ndims))
  shape = combined_static_and_dynamic_shape(inputs)
  flattened_dim_prod = tf.reduce_prod(shape[first:last],
                                      keepdims=True)
  new_shape = tf.concat([shape[:first], flattened_dim_prod,
                         shape[last:]], axis=0)
  return tf.reshape(inputs, new_shape)


def flatten_first_n_dimensions(inputs, n):
  """Flattens `K-d` tensor along first n dimension to be a `(K-n+1)-d` tensor"""
  return flatten_dimensions(inputs, first=0, last=n)


def expand_first_dimension(inputs, dims):
  """Expands `K-d` tensor along first dimension to be a `(K+n-1)-d` tensor"""
  inputs_shape = combined_static_and_dynamic_shape(inputs)
  expanded_shape = tf.stack(dims + inputs_shape[1:])

  # Verify that it is possible to expand the first axis of inputs.
  assert_op = tf.assert_equal(
      inputs_shape[0], tf.reduce_prod(tf.stack(dims)),
      message=('First dimension of `inputs` cannot be expanded into provided '
               '`dims`'))

  with tf.control_dependencies([assert_op]):
    inputs_reshaped = tf.reshape(inputs, expanded_shape)

  return inputs_reshaped


def resize_images_and_return_shapes(inputs, image_resizer_fn):
  """Resizes images using the given function and returns their true shapes"""

  if inputs.dtype is not tf.float32:
    raise ValueError('`resize_images_and_return_shapes` expects a'
                     ' tf.float32 tensor')

  
  outputs = static_or_dynamic_map_fn(
      image_resizer_fn,
      elems=inputs,
      dtype=[tf.float32, tf.int32])
  resized_inputs = outputs[0]
  true_image_shapes = outputs[1]

  return resized_inputs, true_image_shapes
