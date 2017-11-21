from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from numpy import prod, ones
from tensorflow.contrib.keras.python.keras.layers import Layer
from tensorflow.contrib.keras.python.keras import initializers
from functools import reduce
from initializers import Identity, TilingInitializer


def get_tensor_shape(t):
    return [-1 if elem is None else elem for elem in t.shape.as_list()]


class _LocalWeightSharing(base.Layer):
    """Abstract nD convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, no bias will
        be applied.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Regularizer function for the output.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: A string, the name of the layer.
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 centroids=3,
                 local_normalization=True,
                 centroids_trainable=True,
                 gain=1.0,
                 per_filter=True,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(_LocalWeightSharing, self).__init__(trainable=trainable,
                                                  name=name, **kwargs)
        self.rank = rank
        self.filters = filters
        self.centroids = centroids
        self.per_filter = per_filter
        self.local_normalization = local_normalization
        self.centroids_trainable = centroids_trainable
        self.gain = gain
        self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = utils.normalize_tuple(strides, rank, 'strides')
        self.padding = utils.normalize_padding(padding)
        self.data_format = utils.normalize_data_format(data_format)
        self.dilation_rate = utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.input_spec = base.InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis].value
        kernel_shape = self.kernel_size + (input_dim, self.filters * self.centroids)

        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=True,
                                        dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_variable(name='bias',
                                          shape=(self.filters * self.centroids,),
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True,
                                          dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                         axes={channel_axis: input_dim})

        kernel_axis = self.rank + 1 if self.data_format == 'channels_last' else 1
        centroid_axis = self.rank + 2
        centroid_broadcasting_shape = ones(self.rank + 3)
        centroid_broadcasting_shape[kernel_axis] = self.filters if self.per_filter else 1
        centroid_broadcasting_shape[centroid_axis] = self.centroids
        self.centroid_coordinates = [
            self.add_variable(
                name='centroids_axis{}'.format(i),
                shape=tuple(centroid_broadcasting_shape),
                initializer=init_ops.random_uniform_initializer(),
                trainable=self.centroids_trainable,
                dtype=self.dtype
            ) for i in range(self.rank)
        ]

        distance_matrix_init_shape = list(centroid_broadcasting_shape) + [self.rank, self.rank]
        self.distance_matrix = self.add_variable(
            name='distance_matrix',
            shape=distance_matrix_init_shape,
            initializer=Identity(gain=self.gain),
            trainable=self.centroids_trainable,
            dtype=self.dtype
        )
        self.built = True

    def call(self, inputs):
        outputs = nn.convolution(
            input=inputs,
            filter=self.kernel,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format, self.rank + 2))

        if self.bias is not None:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    outputs_4d = array_ops.reshape(outputs,
                                                   [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] * outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        outputs = self._linear_local_weight_sharing(outputs)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _linear_local_weight_sharing(self, preactivation):
        kernel_axis = self.rank + 1
        centroid_axis = self.rank + 2
        preactivation_shape = get_tensor_shape(preactivation)
        outputs_centroids_stacked = array_ops.reshape(
            preactivation, preactivation_shape[:kernel_axis] + [self.filters, self.centroids]
        )

        similarities = self._compute_similarities(centroid_axis, preactivation_shape)
        if self.local_normalization:
            local_sums = math_ops.reduce_sum(similarities, axis=centroid_axis, keep_dims=True)
            similarities = math_ops.divide(similarities, local_sums)
        similarity_weighted = math_ops.multiply(outputs_centroids_stacked, similarities)
        outputs = math_ops.reduce_sum(similarity_weighted, axis=centroid_axis)
        return outputs

    def _compute_similarities(self, centroid_axis, preactivation_shape):
        mesh_coordinates = self._init_mesh(preactivation_shape)
        sum_squared_difference = self._compute_weighted_squared_difference(
            centroid_axis, mesh_coordinates, preactivation_shape
        )
        similarities = math_ops.exp(math_ops.negative(sum_squared_difference))
        return similarities

    def _compute_weighted_squared_difference(self, centroid_axis, mesh_coordinates, preactivation_shape):
        difference = array_ops.stack(
            [c - m for c, m in zip(self.centroid_coordinates, mesh_coordinates)],
            axis=centroid_axis + 1
        )
        difference = array_ops.expand_dims(difference, axis=centroid_axis + 1)
        distance_matrix = array_ops.tile(
            self.distance_matrix,
            [1 if (i == 0 or i > 2) else preactivation_shape[i]
             for i in range(len(get_tensor_shape(self.distance_matrix)))]
        )
        difference_weighted = math_ops.matmul(difference, distance_matrix)
        perm = list(range(len(get_tensor_shape(difference_weighted))))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        sum_squared_difference = array_ops.reshape(
            math_ops.matmul(difference_weighted, array_ops.transpose(difference_weighted, perm=perm)),
            get_tensor_shape(difference)[:-2]
        )
        return sum_squared_difference

    def _init_mesh(self, preactivation_shape):
        mesh_coordinates = [
            array_ops.tile(
                array_ops.reshape(
                    math_ops.linspace(0.0, 1.0, preactivation_shape[i + 1]),
                    [-1 if j == i + 1 else 1 for j in range(self.rank + 3)]
                ),
                [1 if (j in [0, i + 1] or j > self.rank) else preactivation_shape[j] for j in range(self.rank + 3)]
            )
            for i in range(self.rank)
        ]
        return mesh_coordinates

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                            new_space)


class LocalWeightSharing2D(_LocalWeightSharing, Layer):
  """2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               centroids=3,
               centroids_trainable=True,
               local_normalization=True,
               per_filter=True,
               gain=1.0,
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=TilingInitializer(
                   inner_initializer='glorot_uniform',
                   axis=3,
                   splits=3
               ),
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(LocalWeightSharing2D, self).__init__(
        rank=2,
        filters=filters,
        centroids=centroids,
        centroids_trainable=centroids_trainable,
        per_filter=per_filter,
        gain=gain,
        local_normalization=local_normalization,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name, **kwargs)


