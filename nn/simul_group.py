import tensorflow as tf
import numpy as np
import scipy, os, pickle, gflags, sys, signal, threading, time, logging
from scipy.io.wavfile import read as wav_read

# Import from file
from utility import *
sys.path.insert(1, './feat/')
from stft import stft, istft

# Set logging
logging.getLogger().setLevel(logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TODO Change arguments before experiments
gflags.DEFINE_string('data_folder','/home/ubuntu/data/wsj0_2mix','Path to wsj0-2mix data set')
gflags.DEFINE_string('wav_list_folder','/home/ubuntu/data/wsj0_2mix','Folder that stores wsj0-2mix wav list')
gflags.DEFINE_integer('is_adam', 1, 'Whether to use adam optimizer')
gflags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate')
gflags.DEFINE_float('lr_decay', 0.84, 'Learning rate decay') 
gflags.DEFINE_float('keep_prob', 0.5, 'Keep rate in dropout')
gflags.DEFINE_integer('epoch', 200, '# of epochs')
gflags.DEFINE_integer('feat_dim', 129, 'Dimension of input feature')
gflags.DEFINE_integer('fft_size', 256, '')
gflags.DEFINE_integer('hop_size', 64, '')
gflags.DEFINE_integer('batch_size', 1, 'Batch size. Each sample in a batch is a whole utterance.') 
gflags.DEFINE_integer('n_layers', 4, '# of downsampling layers')
gflags.DEFINE_integer('n_channels', 64, '# of channels in dense block')
gflags.DEFINE_integer('is_deploy', 0, 'Inference or no. 0: training, 1: inference, 2: generate feature for the next stage')
gflags.DEFINE_string('resume_model', '', 'Model prefix to resume')
gflags.DEFINE_string('exp_name', 'deep_casa_wsj', 'exp name')
gflags.DEFINE_string('time_stamp','test','Name for this run. If not specified, then it is an inference run.')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

# Define variables
feat_dim = FLAGS.feat_dim
n_layers = FLAGS.n_layers
n_channels = FLAGS.n_channels

# Define data and feature folder (wsj0-2mix)
data_folder = FLAGS.data_folder
wav_list_folder = FLAGS.wav_list_folder
base_folder=os.getcwd()+ '/exp/'+FLAGS.exp_name+'/feat'

# wav list
wav_list_prefix = wav_list_folder + '/mix_2_spk_min'
wav_list_tr = wav_list_prefix + '_tr_mix'
wav_list_cv = wav_list_prefix + '_cv_mix'
wav_list_tt = wav_list_prefix + '_tt_mix'

# Total training samples
total_tr_files = len(open(wav_list_tr).readlines(  ))

# Parameters
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size
# Display training loss every 'display_step' batches
display_step = 400

# Compute dimension of padded input feature and output feature
input_shape_tmp, output_shape_tmp, crop_diff_list = getUnetPadding(np.array([1000, feat_dim]))
n_input = input_shape_tmp[1] 
n_output = output_shape_tmp[1]

# Definition of the network
# Define variables for training
xr_batch = tf.placeholder("float", [None, n_input]) # real stft of mixture
xi_batch = tf.placeholder("float", [None, n_input]) # imag stft of mixture
y1r_batch = tf.placeholder("float", [None, n_output]) # real stft of speaker1
y1i_batch = tf.placeholder("float", [None, n_output]) # imag stft of speaker1
y2r_batch = tf.placeholder("float", [None, n_output]) # real stft of  speaker2
y2i_batch = tf.placeholder("float", [None, n_output]) # imag stft of speaker2
sig_batch = tf.placeholder("float", [None, 2])  # waveform of the two speakers
seq_len_batch = tf.placeholder("int32", [None]) # effective length of utterances in a batch (frames)
lr = tf.placeholder("float")
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder("float")

# Define tf queue for training data loading
q = tf.PaddingFIFOQueue(50, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32], shapes=[[None, n_input], [None, n_input], [None, n_output], [None, n_output], [None, n_output], [None, n_output], [None, 2], [None]])
enqueue_op = q.enqueue([xr_batch, xi_batch, y1r_batch, y1i_batch, y2r_batch, y2i_batch, sig_batch, seq_len_batch])
queue_size = q.size()

# Define dequeue operation without predefined batch size
xr_, xi_, y1r_, y1i_, y2r_, y2i_, sig_, seq_len_ = q.dequeue()
xr = tf.placeholder_with_default(xr_, [None, n_input])
xi = tf.placeholder_with_default(xi_, [None, n_input])
y1r = tf.placeholder_with_default(y1r_, [None, n_output])
y1i = tf.placeholder_with_default(y1i_, [None, n_output])
y2r = tf.placeholder_with_default(y2r_, [None, n_output])
y2i = tf.placeholder_with_default(y2i_, [None, n_output])
sig = tf.placeholder_with_default(sig_, [None, 2])
seq_len = tf.placeholder_with_default(seq_len_, [None])

# Function for a new thread to load and enqueue training data
def load_and_enqueue(sess, enqueue_op, coord, queue_index, total_queues):
    all_training_files = list()
    with open(wav_list_tr, 'r') as f:
        for file,line in enumerate(f):
            line = line.split('\n')[0]
            all_training_files.append(line)

    block_size = len(all_training_files)/total_queues
    all_training_files = all_training_files[block_size*queue_index:block_size*(queue_index+1)]

    # file pointer for reading thread
    current_file = 0

    # If not stop, keep reading
    while not coord.should_stop():
        if current_file == len(all_training_files):
            current_file = 0

        if current_file == 0:
            np.random.shuffle(all_training_files)

        # Data to be loaded
        xr_2batch = list()
        xi_2batch = list()
        y1r_2batch = list()
        y1i_2batch = list()
        y2r_2batch = list()
        y2i_2batch = list()
        sig_2batch = list()
        time_list = np.zeros([batch_size]).astype('int32')

        # Load each utterance in the batch
        for batch_index in range(batch_size):
            y1_y2_x_utt = np.load(base_folder+'/tr/'+all_training_files[current_file+batch_index]+'.npy')
            sig_all_utt = np.load(base_folder+'/tr/'+all_training_files[current_file+batch_index]+'_wave.npy')

            # Load each part of the data
            y1r_2batch.append(np.squeeze(y1_y2_x_utt[0,:,:]))
            y1i_2batch.append(np.squeeze(y1_y2_x_utt[1,:,:]))
            y2r_2batch.append(np.squeeze(y1_y2_x_utt[2,:,:]))
            y2i_2batch.append(np.squeeze(y1_y2_x_utt[3,:,:]))
            xr_2batch.append(np.squeeze(y1_y2_x_utt[4,:,:]))
            xi_2batch.append(np.squeeze(y1_y2_x_utt[5,:,:]))
            sig_2batch.append(np.transpose(sig_all_utt[:2,:]))

            time_list[batch_index] = y1_y2_x_utt.shape[1]
           
        # length of the longest utterance in a batch
        max_time = np.max(time_list)
        freq_dim = xr_2batch[0].shape[1]
        # Compute input and output shape for all samples in a batch
        input_shape, output_shape, _ = getUnetPadding(np.array([max_time, freq_dim]))

        # Apply paddings to all training samples
        for batch_index in range(batch_size):
            # Pad all samples in the batch. Pad difference between max_time and output shape (along time axis)
            y1r_2batch[batch_index] = np.pad(y1r_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            y1i_2batch[batch_index] = np.pad(y1i_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            y2r_2batch[batch_index] = np.pad(y2r_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            y2i_2batch[batch_index] = np.pad(y2i_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            xr_2batch[batch_index] = np.pad(xr_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            xi_2batch[batch_index] = np.pad(xi_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            sig_2batch[batch_index] = np.pad(sig_2batch[batch_index], [(FLAGS.hop_size*(output_shape[0]-max_time), FLAGS.hop_size*(max_time-time_list[batch_index])), (0,0)], mode='constant', constant_values=0.0)
            # Pad input time (input is larger than ouput)
            padding_frames = (input_shape[0] - output_shape[0]) / 2
            xr_2batch[batch_index] = np.pad(xr_2batch[batch_index], [(padding_frames, padding_frames), (0,0)], mode='constant', constant_values=0.0) # Pad along time axis
            xi_2batch[batch_index] = np.pad(xi_2batch[batch_index], [(padding_frames, padding_frames), (0,0)], mode='constant', constant_values=0.0) # Pad along time axis
            # Pad frequency
            y1r_2batch[batch_index] = pad_freqs(y1r_2batch[batch_index], output_shape)
            y1i_2batch[batch_index] = pad_freqs(y1i_2batch[batch_index], output_shape)
            y2r_2batch[batch_index] = pad_freqs(y2r_2batch[batch_index], output_shape)
            y2i_2batch[batch_index] = pad_freqs(y2i_2batch[batch_index], output_shape)
            xr_2batch[batch_index] = pad_freqs(xr_2batch[batch_index], input_shape)
            xi_2batch[batch_index] = pad_freqs(xi_2batch[batch_index], input_shape)

        y1r_2batch = np.vstack(y1r_2batch)
        y1i_2batch = np.vstack(y1i_2batch)
        y2r_2batch = np.vstack(y2r_2batch)
        y2i_2batch = np.vstack(y2i_2batch)
        xr_2batch = np.vstack(xr_2batch)
        xi_2batch = np.vstack(xi_2batch)
        sig_2batch = np.vstack(sig_2batch)

        # enqueue operation
        sess.run(enqueue_op, feed_dict={xr_batch: xr_2batch, xi_batch: xi_2batch, y1r_batch: y1r_2batch, y1i_batch: y1i_2batch, y2r_batch: y2r_2batch, y2i_batch: y2i_2batch, sig_batch: sig_2batch, seq_len_batch: time_list})

        current_file += batch_size

# Define Dense_UNet model
def Dense_UNet(_XR, _XI, _seq_len, training, keep_prob):
    # input shape: (batch_size * time, n_input)
    num_seq = tf.shape(_seq_len)[0]
    inputR = _XR
    inputR = tf.reshape(inputR, [num_seq, -1, n_input, 1]) 
    inputR = tf.transpose(inputR, [0,2,1,3])

    inputI = _XI
    inputI = tf.reshape(inputI, [num_seq, -1, n_input, 1]) 
    inputI = tf.transpose(inputI, [0,2,1,3])

    with tf.variable_scope("separator"):

        enc_outputs = list() # list for skip connections
        _initializer=tf.truncated_normal_initializer(stddev=0.02) # Weight initializer for cnn layers
        _initializer_stride = tf.random_normal_initializer(mean=0.0, stddev=0.5) # Weight initializer for downsampling filters

        # Initial input (Batch, frequency, time, channel)
        current_layer = tf.concat([inputR,inputI], 3)

        # First dense block
        for j in range(5):
            if j == 0:
                skip_input = current_layer
            else:
                skip_input = tf.concat([skip_input, current_layer], 3)
            if j == 2:
                # Frequency mapping layer
                current_layer = tf.layers.conv2d(skip_input, n_channels, 1, activation=None, padding='same', kernel_initializer=_initializer)
                current_layer = tf.nn.elu(current_layer)
                current_layer = tf.contrib.layers.layer_norm(current_layer)
                current_layer = tf.transpose(current_layer, [0,3,2,1])
                f_dim = current_layer.get_shape()[-1].value
                current_layer = tf.layers.conv2d(current_layer, f_dim, (1,1), activation=None, padding='same', kernel_initializer=_initializer)
                current_layer = tf.transpose(current_layer, [0,3,2,1])
            elif j != 4:
                current_layer = tf.layers.conv2d(skip_input, n_channels, 3, activation=None, padding='same', kernel_initializer=_initializer)
            else:
                current_layer = tf.layers.conv2d(skip_input, n_channels, 3, activation=None, padding='valid', kernel_initializer=_initializer)
            current_layer = tf.nn.elu(current_layer)
            current_layer = tf.contrib.layers.layer_norm(current_layer)

        enc_outputs.append(current_layer)

        # Downsampling dense blocks
        for i in range(n_layers):
            # Downsampling filtering
            filter = tf.get_variable("w%d"%i, [2, 2, n_channels, 1],initializer=_initializer_stride)
            current_layer = tf.nn.dropout(current_layer, keep_prob)
            current_layer = tf.nn.depthwise_conv2d(current_layer, filter, strides=[1,2,2,1], padding='VALID')
            # Dense block
            for j in range(5):
                if j == 0:
                    skip_input = current_layer
                else:
                    skip_input = tf.concat([skip_input, current_layer], 3)
                if j == 2:
                    current_layer = tf.layers.conv2d(skip_input, n_channels, 1, activation=None, padding='same', kernel_initializer=_initializer)
                    current_layer = tf.nn.elu(current_layer)
                    current_layer = tf.contrib.layers.layer_norm(current_layer)
                    current_layer = tf.transpose(current_layer, [0,3,2,1])
                    f_dim = current_layer.get_shape()[-1].value
                    current_layer = tf.layers.conv2d(current_layer, f_dim, (1,1), activation=None, padding='same', kernel_initializer=_initializer)
                    current_layer = tf.transpose(current_layer, [0,3,2,1])
                elif j != 4:
                    current_layer = tf.layers.conv2d(skip_input, n_channels, 3, activation=None, padding='same', kernel_initializer=_initializer)
                else:
                    current_layer = tf.layers.conv2d(skip_input, n_channels, 3, activation=None, padding='valid', kernel_initializer=_initializer)
                current_layer = tf.nn.elu(current_layer)
                current_layer = tf.contrib.layers.layer_norm(current_layer)
            # Skip connection
            if i < n_layers - 1:
                enc_outputs.append(current_layer)

        # Upsampling dense blocks
        for i in range(n_layers):
            # Transpose convolution
            current_layer = tf.layers.conv2d_transpose(current_layer, n_channels, 2, strides=2, activation=None, padding='valid', kernel_initializer=_initializer)
            current_layer = tf.nn.elu(current_layer)
            current_layer = tf.contrib.layers.layer_norm(current_layer)
            # Skip connnection
            current_layer = crop_and_concat(enc_outputs[-i-1], current_layer, crop_diff_list[i])
            # Dense block
            for j in range(5):
                if j == 0:
                    skip_input = current_layer
                else:
                    skip_input = tf.concat([skip_input, current_layer], 3)
                if j == 2:
                    current_layer = tf.layers.conv2d(skip_input, n_channels, 1, activation=None, padding='same', kernel_initializer=_initializer)
                    current_layer = tf.nn.elu(current_layer)
                    current_layer = tf.contrib.layers.layer_norm(current_layer)
                    current_layer = tf.transpose(current_layer, [0,3,2,1])
                    f_dim = current_layer.get_shape()[-1].value
                    current_layer = tf.layers.conv2d(current_layer, f_dim, (1,1), activation=None, padding='same', kernel_initializer=_initializer)
                    current_layer = tf.transpose(current_layer, [0,3,2,1])
                elif j != 4:
                    current_layer = tf.layers.conv2d(skip_input, n_channels, 3, activation=None, padding='same', kernel_initializer=_initializer)
                else:
                    current_layer = tf.layers.conv2d(skip_input, n_channels, 3, activation=None, padding='valid', kernel_initializer=_initializer)

                current_layer = tf.nn.elu(current_layer)
                current_layer = tf.contrib.layers.layer_norm(current_layer)

        # Output layer
        current_layer = tf.layers.conv2d(current_layer, n_channels, 1, activation=tf.nn.elu, padding='same')

        # Mask estimation, real (R) and imaginary (I)
        mask1R = tf.layers.conv2d(current_layer, 1, 1, activation=None, padding='valid') 
        mask1I = tf.layers.conv2d(current_layer, 1, 1, activation=None, padding='valid')
        mask2R = tf.layers.conv2d(current_layer, 1, 1, activation=None, padding='valid')
        mask2I = tf.layers.conv2d(current_layer, 1, 1, activation=None, padding='valid')

        # Mask range mapping
        mask1R = uncompress(mask1R)
        mask1I = uncompress(mask1I)
        mask2R = uncompress(mask2R)
        mask2I = uncompress(mask2I)

        # Cropped input
        inputR_crop = crop(inputR, crop_diff_list[-1])
        inputI_crop = crop(inputI, crop_diff_list[-1])

        # Complex masking
        est1R = tf.multiply(inputR_crop, mask1R) - tf.multiply(inputI_crop, mask1I)
        est1I = tf.multiply(inputR_crop, mask1I) + tf.multiply(inputI_crop, mask1R)
        est2R = tf.multiply(inputR_crop, mask2R) - tf.multiply(inputI_crop, mask2I)
        est2I = tf.multiply(inputR_crop, mask2I) + tf.multiply(inputI_crop, mask2R)

        # Final R/I estimation
        est1 = tf.concat([est1R,est1I],3)
        est2 = tf.concat([est2R,est2I],3)
        est1 = tf.transpose(est1, [0, 2, 1, 3])
        est2 = tf.transpose(est2, [0, 2, 1, 3])
        est1 = tf.reshape(est1, [-1, 2*n_output])
        est2 = tf.reshape(est2, [-1, 2*n_output])

        # Final mask estimation
        mask1 = tf.concat([mask1R,mask1I],3)
        mask2 = tf.concat([mask2R,mask2I],3)
        mask1 = tf.transpose(mask1, [0, 2, 1, 3])
        mask2 = tf.transpose(mask2, [0, 2, 1, 3])
        mask1 = tf.reshape(mask1, [-1, 2*n_output])
        mask2 = tf.reshape(mask2, [-1, 2*n_output])

    return est1,est2,mask1,mask2

# Define the PIT cost
def PIT_cost(Y1R,Y1I,Y2R,Y2I,EST1,EST2,SIG,_seq_len):

    # Reshape the targets of output
    Y1 = tf.concat([tf.expand_dims(Y1R,-1),tf.expand_dims(Y1I,-1)],2)
    Y1 = tf.reshape(Y1, [-1, 2*n_output])
    Y2 = tf.concat([tf.expand_dims(Y2R,-1),tf.expand_dims(Y2I,-1)],2)
    Y2 = tf.reshape(Y2, [-1, 2*n_output])

    # Calculate loss
    # loss possibility one 
    loss11 = tf.reduce_mean(tf.abs(EST1-Y1), 1)
    loss22 = tf.reduce_mean(tf.abs(EST2-Y2), 1)
    loss1 = tf.add(loss11,loss22)
    # loss possibility two
    loss21 = tf.reduce_mean(tf.abs(EST2-Y1), 1)
    loss12 = tf.reduce_mean(tf.abs(EST1-Y2), 1)
    loss2 = tf.add(loss21,loss12)

    # Get the minimum loss and output assignment
    loss1_tmp = tf.reshape(loss1, [-1,1])
    loss2_tmp = tf.reshape(loss2, [-1,1])
    loss_tmp = tf.concat([loss1_tmp, loss2_tmp], 1)

    min_index = tf.argmin(loss_tmp,axis=1)
    min_index = tf.cast(min_index, tf.float32)
    min_index = tf.reshape(min_index, [-1,1])
    min_index = 1.0 - min_index

    index_11 = (min_index)
    index_12 = (1.0-min_index)
    assign = tf.concat([index_11,index_12],axis=1)
    assign = tf.reshape(assign, [-1,1,2])

    # Reassign the estimates w.r.t. the targets
    EST = tf.concat([tf.expand_dims(EST1,-1), tf.expand_dims(EST2,-1)], 2)
    EST1_reassign = tf.reduce_sum(tf.multiply(EST, assign),2)
    EST2_reassign = tf.reduce_sum(tf.multiply(EST, 1.0-assign),2)

    EST1_reassign = tf.reshape(EST1_reassign, [-1,n_output,2])
    EST2_reassign = tf.reshape(EST2_reassign, [-1,n_output,2])

    # Throw away the last dimension in frequency (padded for cnn)
    EST1_reassign_crop = EST1_reassign[:,:-1,:]
    EST2_reassign_crop = EST2_reassign[:,:-1,:]

    # Complex STFT
    EST1_complex = tf.complex(EST1_reassign_crop[:,:,0],EST1_reassign_crop[:,:,1])
    EST2_complex = tf.complex(EST2_reassign_crop[:,:,0],EST2_reassign_crop[:,:,1])

    num_seq = tf.shape(_seq_len)[0]
    EST1_complex = tf.reshape(EST1_complex, [num_seq, -1, n_output-1])
    EST2_complex = tf.reshape(EST2_complex, [num_seq, -1, n_output-1])

    # iSTFT -> Final time-domain estimates
    EST1_sig = inverse_stft(EST1_complex)
    EST2_sig = inverse_stft(EST2_complex)

    # Two time-domain targets
    Y1_sig = SIG[:,0]
    Y2_sig = SIG[:,1]
    Y1_sig = tf.reshape(Y1_sig, [num_seq, -1])
    Y2_sig = tf.reshape(Y2_sig, [num_seq, -1])

    # Calculate SNR loss
    S_target_1 = Y1_sig
    S_noise_1 = EST1_sig - S_target_1
    SNR1 = 10*log10(tf.reduce_sum(S_target_1*S_target_1, 1)/tf.reduce_sum(S_noise_1*S_noise_1, 1))

    S_target_2 = Y2_sig
    S_noise_2 = EST2_sig - S_target_2
    SNR2 = 10*log10(tf.reduce_sum(S_target_2*S_target_2, 1)/tf.reduce_sum(S_noise_2*S_noise_2, 1))

    SNR = (SNR1 + SNR2)/2
    SNR = tf.reduce_mean(SNR)

    # Calculate complex STFT domain loss
    loss_tmp = tf.minimum(loss1,loss2)
    loss = tf.reduce_mean(loss_tmp)
    loss = loss * tf.reduce_max(tf.to_float(_seq_len)) / tf.reduce_mean(tf.to_float(_seq_len))

    return SNR, loss, EST1_sig, EST2_sig

# Outputs of the network and loss
est1,est2,mask1,mask2=Dense_UNet(xr,xi,seq_len,is_training,keep_prob)
snr,cost,est1_sig,est2_sig=PIT_cost(y1r,y1i,y2r,y2i,est1,est2,sig,seq_len)

# Define loss and optimizer
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    if FLAGS.is_adam == 1:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(-snr) # Adam Optimizer
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(-snr) # Momentum Optimizer

# Print the total number of parameters
separator_vars = [v for v in tf.trainable_variables()]
print("Sep_Vars: " + str(getNumParams(separator_vars)))

# Model dir and model saver
model_dir = os.getcwd()+"/exp/"+FLAGS.exp_name+"/models/"+FLAGS.time_stamp
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
saver = tf.train.Saver(max_to_keep = None) 

# Launch the graph
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if FLAGS.resume_model:
        logging.info('restoring from '+FLAGS.resume_model)
        saver.restore(sess, FLAGS.resume_model)
        logging.info('finished loading checkpoint.')
    else:
        logging.info('training from scratch.')

    # If training, initialize all data loaders
    if FLAGS.is_deploy == 0:
        coord = tf.train.Coordinator()
        num_threads = 2 # Use 2 threads for data loading, each thread touches different part of training data
        t = [threading.Thread(target=load_and_enqueue, args=(sess,enqueue_op,coord,i,num_threads))  for i in xrange(num_threads)]
        for tmp_t in t:
            tmp_t.start()
    
    # Use pickle file to save temporary training specs
    training_info_pickle = model_dir+'/training_info.pickle'
    if os.path.isfile(training_info_pickle):
        # If there is a pickle file, then load to restore previous training states
        with open(training_info_pickle) as f_pickle:
            step, best_cv_snr, best_cv_step, learning_rate = pickle.load(f_pickle)
        saver.restore(sess, os.getcwd()+"/exp/"+FLAGS.exp_name+"/models/"+FLAGS.time_stamp+'/'+FLAGS.exp_name+"_model.ckpt" + "_step_"+ str(step))    
    else:
        # If no pickle file, then initialize training specs
        step = 1
        best_cv_snr = -100
        best_cv_step = 0

    # Training cost and snr
    train_cost_sum = 0
    train_snr_sum = 0
    total_num = 0
    # CV data reader
    cv_buffer = list()
    cv_sig_buffer = list()
    cv_loaded = 0

    # Keep training until reaching max iterations
    while step  < FLAGS.epoch * total_tr_files / batch_size:
        
        # At the beginning of each epoch
        # If FLAGS.is_deploy is 0, run validation at the beginning of each epoch
        # IF FLAGS.is_deploy is 1, run evaluation and exit
        # If FLAGS.is_deploy is 2, generate data for sequential grouping and exit
        if step % (total_tr_files / batch_size) == 1:
            
            if FLAGS.is_deploy == 0:
                # CV
                logging.info('Start to CV')
                cost_list = list() 
                snr_list = list()
                with open(wav_list_cv, 'r') as f:
                    for file,line in enumerate(f):
                        if file>=500: # TODO If want to skip some cv files
                            break
                        line = line.split('\n')[0]
                        # Load cv files
                        if cv_loaded == 0:
                            # If not loaded, load from file system
                            y1_y2_x_utt = np.load(base_folder+'/cv/'+line+'.npy')
                            sig_all_utt = np.load(base_folder+'/cv/'+line+'_wave.npy')
                            cv_buffer.append(y1_y2_x_utt)
                            cv_sig_buffer.append(sig_all_utt)
                        else:
                            # If loaded, load from buffer
                            y1_y2_x_utt = cv_buffer[file]
                            sig_all_utt = cv_sig_buffer[file]

                        # load features and targets
                        y1r_utt, y1i_utt, y2r_utt, y2i_utt, xr_utt, xi_utt, sig_utt, utt_len, time_tmp, y1r_utt_pad, y1i_utt_pad, y2r_utt_pad, y2i_utt_pad, xr_utt_pad, xi_utt_pad, sig_utt_pad = single_utt_feature_proc(y1_y2_x_utt, sig_all_utt) 

                        test_cost, test_snr = sess.run([cost, snr], \
                        feed_dict={xr: xr_utt_pad, xi: xi_utt_pad, y1r: y1r_utt_pad, y1i: y1i_utt_pad, y2r: y2r_utt_pad, y2i: y2i_utt_pad, sig: sig_utt_pad, seq_len: utt_len, keep_prob: 1, is_training: False})
                        cost_list.append(test_cost)
                        snr_list.append(test_snr)
                        
                # Set flag to 1
                cv_loaded = 1

                # Calculate mean loss and snr
                mean_cost = np.mean(cost_list[:])
                mean_snr = np.mean(snr_list[:])
                
                # If best cv snr, update the corresponding variable
                if mean_snr >= best_cv_snr:
                    best_cv_snr = mean_snr
                    best_cv_step = step
                    logging.info('Best cv snr !!!!!!!!!!!!!!!!!!!') 
                
                # Output cv loss and snr
                logging.info('cv_cost %f',mean_cost)
                logging.info('cv_snr %f',mean_snr)
            else:
                # Either generate .npy file for sequential grouping training (all tr, cv, and tt), or generate file for evaluation (tt only)
                if FLAGS.is_deploy == 1:
                    task_list = ["tt"]
                elif FLAGS.is_deploy == 2:
                    task_list = ["tr","cv","tt"]

                for tasks in range(len(task_list)):
                    wav_list = wav_list_prefix + '_' + task_list[tasks] + '_mix'

                    with open(wav_list, 'r') as f:
                        for file,line in enumerate(f):

                            line = line.split('\n')[0]
                            y1_y2_x_utt = np.load(base_folder+'/'+task_list[tasks]+'/'+line+'.npy')
                            sig_all_utt = np.load(base_folder+'/'+task_list[tasks]+'/'+line+'_wave.npy')

                            # Load features and targets
                            y1r_utt, y1i_utt, y2r_utt, y2i_utt, xr_utt, xi_utt, sig_utt, utt_len, time_tmp, y1r_utt_pad, y1i_utt_pad, y2r_utt_pad, y2i_utt_pad, xr_utt_pad, xi_utt_pad, sig_utt_pad = single_utt_feature_proc(y1_y2_x_utt, sig_all_utt)
                            # Either generate .npy file for sequential grouping training, or generate file for evaluation
                            if FLAGS.is_deploy == 2:
                                # .npy file for sequential grouping training
                                est_1, est_2=sess.run([est1,est2],\
                                    feed_dict={xr: xr_utt_pad, xi: xi_utt_pad, y1r: y1r_utt_pad, y1i: y1i_utt_pad, y2r: y2r_utt_pad, y2i: y2i_utt_pad, sig: sig_utt_pad, seq_len: utt_len, keep_prob: 1, is_training: False})
                                total_est = np.stack((est_1,est_2),axis=0)
                                np.save(base_folder+'/'+task_list[tasks]+'/'+line +'_simul_est', total_est)
                                logging.info('saving ' + base_folder+'/'+task_list[tasks]+'/'+line +'_simul_est')

                            elif FLAGS.is_deploy == 1:
                                # Generate file for evaluation
                                est_1, est_2, sig_1, sig_2 =sess.run([est1,est2,est1_sig,est2_sig],\
                                    feed_dict={xr: xr_utt_pad, xi: xi_utt_pad, y1r: y1r_utt_pad, y1i: y1i_utt_pad, y2r: y2r_utt_pad, y2i: y2i_utt_pad, sig: sig_utt_pad, seq_len: utt_len, keep_prob: 1, is_training: False})
                                # Reshape and crop estimates
                                input_shape_test, output_shape_test, _ = getUnetPadding(np.array([xr_utt.shape[0], xr_utt.shape[1]]), n_layers=4)
                                est_1 = np.reshape(est_1, [-1, output_shape_test[1], 2])
                                est_2 = np.reshape(est_2, [-1, output_shape_test[1], 2])
                                est_1 = est_1[(output_shape_test[0]-time_tmp):,:-1,:]
                                est_2 = est_2[(output_shape_test[0]-time_tmp):,:-1,:]
                                est_1 = np.reshape(est_1, (-1, 2*feat_dim))
                                est_2 = np.reshape(est_2, (-1, 2*feat_dim))

                                # Reshaping targets
                                y1_utt = np.concatenate((np.expand_dims(y1r_utt,-1),np.expand_dims(y1i_utt,-1)),2)
                                y1_utt = np.reshape(y1_utt, (-1, 2*feat_dim))
                                y2_utt = np.concatenate((np.expand_dims(y2r_utt,-1),np.expand_dims(y2i_utt,-1)),2)
                                y2_utt = np.reshape(y2_utt, (-1, 2*feat_dim))

                                # Reshape and crop time-domain outputs
                                sig_1 = np.reshape(sig_1, (-1))
                                sig_2 = np.reshape(sig_2, (-1))
                                sig_1 = sig_1[(output_shape_test[0]-time_tmp)*FLAGS.hop_size:]
                                sig_2 = sig_2[(output_shape_test[0]-time_tmp)*FLAGS.hop_size:]

                                # load original wav file
                                wav_folders = data_folder + '/2speakers/wav8k/min/tt/'
                                sr,clean_audio_1 = wav_read(wav_folders+'s1/'+line+'.wav')
                                clean_audio_1 = clean_audio_1.astype('float32')/np.power(2,15)
                                sr,clean_audio_2 = wav_read(wav_folders+'s2/'+line+'.wav')
                                clean_audio_2 = clean_audio_2.astype('float32')/np.power(2,15)
                                sr,mix_audio = wav_read(wav_folders+'mix/'+line+'.wav')
                                mix_audio = mix_audio.astype('float32')/np.power(2,15)

                                # reconstructed two speakers (waveform)
                                res_1 = sig_1.astype('float32')
                                res_2 = sig_2.astype('float32')

                                """
                                # The time domain signal sig_1 and sig_2 should be the same as the signals generated below:
                                # Get optimal assignment
                                assign = opt_assign(y1_utt,y2_utt,est_1,est_2)
                                # Reassign w.r.t. opt assign
                                est_1_tmp = est_1.copy()
                                est_2_tmp = est_2.copy()
                                for i in range(est_1_tmp.shape[0]):
                                    if assign[i] == 0:
                                        est_1[i,:] = est_2_tmp[i,:]
                                        est_2[i,:] = est_1_tmp[i,:]
                                # Get real and imaginary part
                                est_1 = np.reshape(est_1, (-1, feat_dim, 2))
                                est_2 = np.reshape(est_2, (-1, feat_dim, 2))
                                est_1r = np.squeeze(est_1[:,:,0])
                                est_1i = np.squeeze(est_1[:,:,1])
                                est_2r = np.squeeze(est_2[:,:,0])
                                est_2i = np.squeeze(est_2[:,:,1])
                                # Compute time-domain signals
                                RES_1 = est_1r + 1j* est_1i
                                RES_2 = est_2r + 1j* est_2i
                                RES_1 = np.concatenate((RES_1, np.conj(RES_1[:,::-1][:,1:-1])), axis=1)
                                RES_2 = np.concatenate((RES_2, np.conj(RES_2[:,::-1][:,1:-1])), axis=1)
                                res_1 = istft(RES_1,len(clean_audio_1))
                                res_2 = istft(RES_2,len(clean_audio_2))
                                """
                                # Save mixture, clean signals and estimates in the file folder for evaluation.
                                s_res=np.concatenate((res_1.reshape(-1,1),res_2.reshape(-1,1)),1)
                                s_c=np.concatenate((clean_audio_1.reshape(-1,1),clean_audio_2.reshape(-1,1)),1)
                                # Pad or crop according to the clean source
                                if s_res.shape[0]>s_c.shape[0]:
                                    s_res=s_res[:s_c.shape[0],:]
                                else:
                                    s_res=np.concatenate((s_res,np.zeros([s_c.shape[0]-s_res.shape[0],2]).astype('float32')),0)
                                s_res = s_res.transpose()
                                s_mix =np.concatenate((mix_audio.reshape(-1,1),mix_audio.reshape(-1,1)),1)
                                s_mix = s_mix.transpose()
                                s_c = s_c.transpose()
                                # s_concat is a 6xT matrix 
                                # T is the total number of time samples
                                # The first two dims in 6 correspond to clean sources
                                # Next two dims correspond to estimates
                                # Last two dims correspond to the mixture
                                s_concat = np.concatenate((s_c,s_res,s_mix),axis=0)
                                test_file=os.getcwd()+'/exp/'+FLAGS.exp_name+'/output_tt/files/'+line+'.bin'
                                logging.info('saving ' + test_file)
                                s_concat.tofile(test_file)
                                
                sys.exit(0)

            # Saving models and training specs after evaluation
            save_path = saver.save(sess, os.getcwd()+"/exp/"+FLAGS.exp_name+"/models/"+FLAGS.time_stamp+'/'+FLAGS.exp_name+"_model.ckpt" + "_step_"+ str(step))
            with open(training_info_pickle, 'w') as f_pickle:
                pickle.dump([step, best_cv_snr, best_cv_step, learning_rate], f_pickle)
            logging.info('Model saved in file: %s',save_path)
        
        # Apply learning rate decay
        # Check every three epochs, if the cv snr does not increase for more than 6 epochs, apply lr_decay
        if (step % (3*(total_tr_files / batch_size)) == 1) and (step != 1):
            if step - best_cv_step >= int(5.9*(total_tr_files / batch_size)):
                learning_rate *= FLAGS.lr_decay

        # Run one batch of training
        _, train_cost, train_snr, q_size\
        = sess.run([optimizer, cost, snr, queue_size], \
        feed_dict={lr: learning_rate, keep_prob: FLAGS.keep_prob, is_training: True})

        # Update snr and loss
        train_cost_sum += train_cost
        train_snr_sum += train_snr
        total_num += 1

        # If reaches display step, print loss
        if step % display_step == 1:
            logging.info('Step %d/%d, Minibatch Loss=%f, SNR=%f, Q_size=%d, Learning Rate=%f',step, FLAGS.epoch * total_tr_files / batch_size, train_cost_sum/total_num, train_snr_sum/total_num, q_size, learning_rate)
            # Reset all snr and loss variables
            train_cost_sum = 0
            train_snr_sum = 0
            total_num = 0
        
        step += 1

    sys.exit(0)

