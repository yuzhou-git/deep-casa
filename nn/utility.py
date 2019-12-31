import numpy as np
import tensorflow as tf
import scipy.signal
from tensorflow.contrib.signal import overlap_and_add
from tensorflow.python.framework import ops

def getTrainableVariables(tag=""):
    return [v for v in tf.trainable_variables() if tag in v.name]

def getNumParams(tensors):
    return np.sum([np.prod(t.get_shape().as_list()) for t in tensors])

# Optimal assignment of outputs
def opt_assign(y1,y2,est1,est2):
    err1 = np.mean(np.abs(y1-est1)+np.abs(y2-est2),1)
    err2 = np.mean(np.abs(y1-est2)+np.abs(y2-est1),1)
    ass = (np.minimum(err1,err2)==err1).astype('float32')
    return ass

# Compute input/output Dense-UNet paddings for a 2-D input shape
def getUnetPadding(shape, n_layers=4):

    rem = shape 
    # Compute shape of the encoding layer
    for i in range(n_layers):
        rem = rem + 2 # Conv
        if np.sum(rem % 2) > 0:
            rem = np.asarray(rem, dtype=np.float32)
        rem = (rem / 2) #Transposed-up-conv

    # Round resulting feature map dimensions up to nearest EVEN integer (even because up-convolution by factor two is needed)
    inner_shape = np.asarray(np.ceil(rem),dtype=np.int64)

    # Compute input and output shapes based on encoding feature map
    output_shape = inner_shape
    input_shape = inner_shape
    # Shape difference between the input and output feature maps at the same hierarchical level: diff_list
    diff_list = list()
    for i in range(n_layers):
        output_shape = output_shape * 2 - 2
        input_shape = (input_shape + 2) * 2
        diff_list.append(input_shape - (output_shape + 2))

    input_shape += 2 # First conv
    diff_list.append(input_shape - output_shape)

    return input_shape, output_shape, diff_list

# Pad frequency for input 2-D tensor (T,F)
def pad_freqs(tensor, target_shape):

    target_freqs = target_shape[1]
    input_shape = tensor.shape
    input_freqs = input_shape[1]

    diff = target_freqs - input_freqs
    if diff % 2 == 0:
        pad = [(diff/2, diff/2)]
    else:
        pad = [(diff//2, diff//2 + 1)] # Add extra frequency bin at the end

    pad = [(0,0)] + pad

    return np.pad(tensor, pad, mode='constant', constant_values=0.0)

def crop(x1, diff_x1_x2):
    offsets0 = diff_x1_x2[0]//2
    offsets1 = diff_x1_x2[1]//2
    x1 = x1[:,offsets0:-offsets0,offsets1:-offsets1,:]
    return x1

def crop_and_concat(x1,x2,diff_x1_x2):
    x1 = crop(x1,diff_x1_x2)
    return tf.concat([x1, x2], axis=3)

# Map estimated masks to a different range
def uncompress(x):
    x = tf.clip_by_value(x,-9.999, 9.999)
    x = -10.0 * tf.log((10.0-x)/(10.0+x))
    return x

# TF calculation of iSTFT
def inverse_stft(stfts,
                 frame_length=256,
                 frame_step=64,
                 fft_length=256):

    with tf.variable_scope("ISTFT"):

        fft_len = fft_length
        stfts = ops.convert_to_tensor(stfts, name='stfts')
        frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
        frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
        fft_length = ops.convert_to_tensor(fft_length, name='fft_length')
        real_frames = tf.signal.irfft(stfts, [fft_length])

        # Declare the window function.
        window = scipy.signal.hanning(fft_len, False)
        non_zero_value = window[1].astype('float32')
        tf_non_zero = tf.constant(non_zero_value, name="tf_non_zero", dtype=tf.float32)

        window = np.sqrt(window)
        window = window.reshape(1,1,fft_len).astype('float32')
        tf_window = tf.constant(window, name="tf_window", dtype=tf.float32)
        real_frames *= tf_window
        stft_weights = tf.ones_like(real_frames,dtype=tf.float32)
        stft_weights *= (tf_window * tf_window)

        time_sig = overlap_and_add(real_frames, frame_step)
        time_weights = overlap_and_add(stft_weights, frame_step)
        time_weights = tf.maximum(time_weights, non_zero_value)

        time_sig = time_sig /time_weights
        return time_sig

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def parametric_relu(_x, name):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

# Dilated temporal convolution with input tensor shape (batch, time, channel) 
def atrous_depth_conv1d(tensor, output_channels, keep_prob, is_causal=False, rate=1, pad='SAME', stddev=0.02, name="aconv1d"):
    # Set filter size
    size = 3

    # Get input dimension
    in_dim = tensor.get_shape()[-1].value

    with tf.variable_scope(name):
        # Make filter
        filter = tf.get_variable("w", [1, size, in_dim, 1],
                                   initializer=tf.truncated_normal_initializer(stddev=stddev))

        random_num_left = tf.random_uniform([1,1],minval=0,maxval=0.9999,dtype=tf.float32,name='left')
        random_num_right = tf.random_uniform([1,1],minval=0,maxval=0.9999,dtype=tf.float32,name='right')
        random_num_middle = tf.random_uniform([1,1],minval=0,maxval=0,dtype=tf.float32,name='middle')

        keep_prob_compare = tf.reshape(keep_prob, [1,1])

        # Form a drop_mask w.r.t. the keep rate
        drop_mask_left = tf.cast(tf.less(random_num_left,keep_prob_compare), tf.float32)
        drop_mask_right = tf.cast(tf.less(random_num_right,keep_prob_compare), tf.float32)
        drop_mask_middle = tf.cast(tf.less(random_num_middle,keep_prob_compare), tf.float32)
        drop_mask = tf.concat([drop_mask_left, drop_mask_middle, drop_mask_right], 1)

        # Multiply the drop to the dilated cnn filter
        filter_new = filter * tf.reshape(drop_mask, [1, size, 1, 1])

        # Pre processing for dilated convolution, expand one dimension
        x = tf.expand_dims(tensor, axis=1)
        # Apply 2d convolution
        out = tf.nn.depthwise_conv2d(x, filter_new, strides=[1,1,1,1], padding=pad, rate=[1, int(rate)])
        out = tf.squeeze(out, axis=1)

    return out

# Generic 1-D convolution
def conv1d(input_, output_channels, filter_width = 1, stride = 1, stddev=0.02, name = 'conv1d'):

    # Get input dimension
    input_shape = input_.get_shape()
    input_channels = input_shape[-1].value

    with tf.variable_scope(name):
        # Make filter
        filter_ = tf.get_variable('w', [filter_width, input_channels, output_channels],
                                   initializer=tf.truncated_normal_initializer(stddev=stddev))
        # Convolution layer
        conv = tf.nn.conv1d(input_, filter_, stride = stride, padding = 'SAME')
        biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))

        # Add bias
        conv = tf.nn.bias_add(conv, biases)

        return conv

# One residual sub-block in a TCN block 
def residual_block(input_, rate,  n_dilated_units, n_bottle, keep_prob, scope="res"):

    input_shape = input_.get_shape()
    input_channels = input_shape[-1].value

    with tf.variable_scope(scope):

        aconv = conv1d(input_,
                            output_channels=n_dilated_units,
                            name="bn_filter")
        aconv = parametric_relu(aconv, "bn_prelu")
        aconv = tf.contrib.layers.layer_norm(aconv)

        aconv = atrous_depth_conv1d(aconv,
                              output_channels=n_dilated_units,
                              keep_prob = keep_prob,
                              rate=rate,
                              name="dilate_filter")
        aconv = parametric_relu(aconv, "dilate_prelu")
        aconv = tf.contrib.layers.layer_norm(aconv)

        aconv = conv1d(aconv,
                            output_channels=n_bottle,
                            name="resyn_filter")

        res_output = input_ 

        return res_output+aconv

# Process validation and evaluation data to feed into the model
def single_utt_feature_proc(y1_y2_x_utt, sig_all_utt, n_layers=4, hop_size=64):
    # load features and targets
    y1r_utt = np.squeeze(y1_y2_x_utt[0,:,:])
    y1i_utt = np.squeeze(y1_y2_x_utt[1,:,:])
    y2r_utt = np.squeeze(y1_y2_x_utt[2,:,:])
    y2i_utt = np.squeeze(y1_y2_x_utt[3,:,:])
    xr_utt = np.squeeze(y1_y2_x_utt[4,:,:])
    xi_utt = np.squeeze(y1_y2_x_utt[5,:,:])
    utt_len = np.asarray([xr_utt.shape[0]]).astype('int32')

    time_tmp = xr_utt.shape[0]
    input_shape_test, output_shape_test, _ = getUnetPadding(np.array([xr_utt.shape[0], xr_utt.shape[1]]), n_layers=4)

    # Pad according to the output size (along time axis)
    y1r_utt_pad = np.pad(y1r_utt, [(output_shape_test[0]-time_tmp, 0), (0,0)], mode='constant', constant_values=0.0)
    y1i_utt_pad = np.pad(y1i_utt, [(output_shape_test[0]-time_tmp, 0), (0,0)], mode='constant', constant_values=0.0)
    y2r_utt_pad = np.pad(y2r_utt, [(output_shape_test[0]-time_tmp, 0), (0,0)], mode='constant', constant_values=0.0)
    y2i_utt_pad = np.pad(y2i_utt, [(output_shape_test[0]-time_tmp, 0), (0,0)], mode='constant', constant_values=0.0)
    xr_utt_pad = np.pad(xr_utt, [(output_shape_test[0]-time_tmp, 0), (0,0)], mode='constant', constant_values=0.0)
    xi_utt_pad = np.pad(xi_utt, [(output_shape_test[0]-time_tmp, 0), (0,0)], mode='constant', constant_values=0.0)
    # Pad input (along time axis)
    padding_frames = (input_shape_test[0] - output_shape_test[0]) / 2
    xr_utt_pad = np.pad(xr_utt_pad, [(padding_frames, padding_frames), (0,0)], mode='constant', constant_values=0.0) # Pad along time axis
    xi_utt_pad = np.pad(xi_utt_pad, [(padding_frames, padding_frames), (0,0)], mode='constant', constant_values=0.0) # Pad along time axis
    # Pad frequency
    y1r_utt_pad = pad_freqs(y1r_utt_pad, output_shape_test)
    y1i_utt_pad = pad_freqs(y1i_utt_pad, output_shape_test)
    y2r_utt_pad = pad_freqs(y2r_utt_pad, output_shape_test)
    y2i_utt_pad = pad_freqs(y2i_utt_pad, output_shape_test)
    xr_utt_pad = pad_freqs(xr_utt_pad, input_shape_test)
    xi_utt_pad = pad_freqs(xi_utt_pad, input_shape_test)

    if sig_all_utt is not None:
        sig_utt = np.transpose(sig_all_utt[:2,:])
        sig_utt_pad = np.pad(sig_utt, [(hop_size*(output_shape_test[0]-time_tmp), 0), (0,0)], mode='constant', constant_values=0.0)
    else:
        sig_utt = None
        sig_utt_pad = None

    return y1r_utt, y1i_utt, y2r_utt, y2i_utt, xr_utt, xi_utt, sig_utt, utt_len, time_tmp, y1r_utt_pad, y1i_utt_pad, y2r_utt_pad, y2i_utt_pad, xr_utt_pad, xi_utt_pad, sig_utt_pad 
