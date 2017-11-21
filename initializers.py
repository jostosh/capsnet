from tensorflow.python.ops.init_ops import Initializer, _assert_float_dtype
from tensorflow.python.ops import linalg_ops, array_ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.keras.python.keras import initializers
from copy import deepcopy


class Identity(Initializer):
    """Initializer that generates the identity matrix.
    Only use for 2D matrices.
    Args:
      gain: Multiplicative factor to apply to the identity matrix.
      dtype: The type of the output.
    """

    def __init__(self, gain=1.0, dtype=dtypes.float32):
        self.gain = gain
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

    def __call__(self, shape, dtype=None, partition_info=None):
        full_shape = shape if partition_info is None else partition_info.full_shape

        if dtype is None:
            dtype = self.dtype
        if len(full_shape) > 2:
            batch_shape = full_shape[:-2]
            rows, cols = full_shape[-2:]
            initializer = linalg_ops.eye(rows, cols, batch_shape=batch_shape, dtype=dtype)
        elif len(full_shape) == 2:
            initializer = linalg_ops.eye(*full_shape, dtype=dtype)
        else:
            raise ValueError(
                "Identity matrix initializer can only be used for shapes with 2 dimensions or more.")
        if partition_info is not None:
            initializer = array_ops.slice(initializer, partition_info.var_offset,
                                          shape)
        return self.gain * initializer

    def get_config(self):
        return {"gain": self.gain, "dtype": self.dtype.name}


class TilingInitializer(Initializer):
    """
    Initializes a tiled tensor where we tile along a single axis
    Args:
        inner_initializer: The initializer instance that initializes a single block
        axis: The axis to tile along
        splits: The number of splits
    """

    def __init__(self, inner_initializer, axis, splits):
        self.inner_initializer = initializers.get(inner_initializer)
        self.axis = axis
        self.splits = splits

    def __call__(self, shape, dtype=None, partition_info=None):
        if shape[self.axis] % self.splits != 0:
            raise ValueError("Axis {} with length {} is not multiple of {}".format(
                self.axis, shape[self.axis], self.splits))
        single_block_shape = deepcopy(shape)
        single_block_shape[self.axis] //= self.splits
        single_block = self.inner_initializer(shape=single_block_shape, dtype=dtype, partition_info=partition_info)
        return array_ops.tile(single_block, [1 if j != self.axis else self.splits for j in range(len(shape))])

    def get_config(self):
        return {"axis": self.axis, "splits": self.splits, "inner_init": self.inner_initializer}


class ConcatInitializer(Initializer):
    """
    Initializes a concatenated tensor where we concatenate along a single axis. This is useful for local weight sharing
    layers, where the kernels of a single kernel-centroid pair should be initialized as if they are the only kernel
    being used at all.
    Args:
        inner_initializer: The initializer instance that initializes a single block
        axis: The axis to tile along
        splits: The number of splits
    """

    def __init__(self, inner_initializer, axis, splits):
        self.inner_initializer = initializers.get(inner_initializer)
        self.axis = axis
        self.splits = splits

    def __call__(self, shape, dtype=None, partition_info=None):
        if shape[self.axis] % self.splits != 0:
            raise ValueError("Axis {} with length {} is not multiple of {}".format(
                self.axis, shape[self.axis], self.splits))
        single_block_shape = deepcopy(shape)
        single_block_shape[self.axis] //= self.splits
        single_blocks = [self.inner_initializer(shape=single_block_shape, dtype=dtype, partition_info=partition_info)
                         for _ in range(self.splits)]
        return array_ops.concat(single_blocks, axis=self.axis)

    def get_config(self):
        return {"axis": self.axis, "splits": self.splits, "inner_init": self.inner_initializer}
