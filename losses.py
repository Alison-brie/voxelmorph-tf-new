import tensorflow as tf
import keras.backend as K
import neuron.layers as nrn_layers
import numpy as np

def mse(x1, x2):
    return tf.losses.mean_squared_error(labels=x1, predictions=x2)

def ncc(I, J):

    I = nrn_layers.Resize(zoom_factor=0.75, interp_method='linear')(I)
    J = nrn_layers.Resize(zoom_factor=0.75, interp_method='linear')(J)

    eps = 1e-5

    ndims = len(I.get_shape().as_list()) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [7] * ndims
    # get convolution function
    conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I * I
    J2 = J * J
    IJ = I * J

    # compute filters
    sum_filt = tf.ones([*win, 1, 1])
    strides = [1] * (ndims + 2)
    padding = 'SAME'

    # compute local sums via convolution
    I_sum = conv_fn(I, sum_filt, strides, padding)
    J_sum = conv_fn(J, sum_filt, strides, padding)
    I2_sum = conv_fn(I2, sum_filt, strides, padding)
    J2_sum = conv_fn(J2, sum_filt, strides, padding)
    IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

    # compute cross correlation
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + eps)

    # return negative cc.
    return -tf.reduce_mean(cc)

def Grad(y, penalty='l2'):

    ndims = 3

    df = [None] * ndims
    for i in range(ndims):
        d = i + 1
        # permute dimensions to put the ith dimension first
        r = [d, *range(d), *range(d + 1, ndims + 2)]
        y = K.permute_dimensions(y, r)
        dfi = y[1:, ...] - y[:-1, ...]

        # permute back
        # note: this might not be necessary for this loss specifically,
        # since the results are just summed over anyway.
        r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
        df[i] = K.permute_dimensions(dfi, r)

    if penalty == 'l2':
        df = [tf.reduce_mean(f * f) for f in df]
    else:
        # assert penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % penalty
        df = [tf.reduce_mean(tf.abs(f)) for f in df]
    return tf.add_n(df) / len(df)

