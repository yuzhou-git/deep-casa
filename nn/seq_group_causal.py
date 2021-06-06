import tensorflow as tf
import numpy as np
import scipy, os, pickle, gflags, sys, signal, threading, time, logging
from scipy.io.wavfile import read as wav_read
import sklearn.cluster as cluster

# Import from file
from utility import *
os.sys.path.insert(0, './feat/')
from stft import stft, istft

# Set logging
logging.getLogger().setLevel(logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TODO Change arguments before experiments
gflags.DEFINE_string('data_folder','/home/ubuntu/data/wsj0_2mix','Path to wsj0-2mix data set')
gflags.DEFINE_string('wav_list_folder','/home/ubuntu/data/wsj0_2mix','Folder that stores wsj0-2mix wav list')
gflags.DEFINE_integer('is_adam', 1, 'Whether to use adam optimizer')
gflags.DEFINE_float('learning_rate', 0.00025, 'Initial learning rate')
gflags.DEFINE_float('lr_decay', 0.93, 'Learning rate decay')
gflags.DEFINE_float('keep_prob', 0.7, 'Keep rate in dropout')
gflags.DEFINE_integer('epoch', 200, '# of epochs')
gflags.DEFINE_integer('feat_dim', 129, 'Dimension of input feature')
gflags.DEFINE_integer('batch_size', 1, 'Batch size. Each sample in a batch is a whole utterance.')
gflags.DEFINE_integer('is_deploy', 0, 'Inference or no. 0: training, 1: inference')
gflags.DEFINE_integer('n_layers', 4, '# of downsampling layers in Dense-UNet')
gflags.DEFINE_integer('n_tcn_block', 4, '# of TCN blocks in TCN')
gflags.DEFINE_integer('n_dilation_each_block', 7, '# of dilated CNN layer in each TCN block')
gflags.DEFINE_integer('n_dilated_units', 512, '# of hidden units in dilated conv in TCN')
gflags.DEFINE_integer('n_bottle', '256', '# of hidden units in bottleneck layers in TCN')
gflags.DEFINE_integer('embedding_dim', 40, 'Dimension of embedding')
gflags.DEFINE_string('resume_model', '', 'Model prefix to resume')
gflags.DEFINE_string('exp_name', 'deep_casa_wsj', 'exp name')
gflags.DEFINE_string('time_stamp','test','Name for this run. If not specified, then it is an inference run.')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

# Define variables
feat_dim = FLAGS.feat_dim
n_layers = FLAGS.n_layers
n_tcn_block = FLAGS.n_tcn_block
n_dilation_each_block = FLAGS.n_dilation_each_block
n_dilated_units = FLAGS.n_dilated_units
n_bottle = FLAGS.n_bottle
embedding_dim = FLAGS.embedding_dim

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

# Compute dimension of padded input feature and output feature in Dense-UNet
input_shape_tmp, output_shape_tmp, crop_diff_list = getUnetPadding(np.array([1000, feat_dim]), is_causal=True)
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
est1_batch = tf.placeholder("float", [None, n_output*2]) # real imag stft of estimate1
est2_batch = tf.placeholder("float", [None, n_output*2]) # real imag stft of estimate2
seq_len_batch = tf.placeholder("int32", [None]) # effective length of utterances in a batch (frames)
lr = tf.placeholder("float")
keep_prob = tf.placeholder("float")
is_training = tf.placeholder(tf.bool)

# Define tf queue for training data loading
q = tf.PaddingFIFOQueue(50, [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32], shapes=[[None, n_input], [None, n_input], [None, n_output], [None, n_output], [None, n_output], [None, n_output], [None, n_output*2], [None, n_output*2], [None]])
enqueue_op = q.enqueue([xr_batch, xi_batch, y1r_batch, y1i_batch, y2r_batch, y2i_batch, est1_batch, est2_batch, seq_len_batch])
queue_size = q.size()

# Define dequeue operation without predefined batch size
xr_, xi_, y1r_, y1i_, y2r_, y2i_, est1_, est2_, seq_len_ = q.dequeue()
xr = tf.placeholder_with_default(xr_, [None, n_input])
xi = tf.placeholder_with_default(xi_, [None, n_input])
y1r = tf.placeholder_with_default(y1r_, [None, n_output])
y1i = tf.placeholder_with_default(y1i_, [None, n_output])
y2r = tf.placeholder_with_default(y2r_, [None, n_output])
y2i = tf.placeholder_with_default(y2i_, [None, n_output])
est1 = tf.placeholder_with_default(est1_, [None, n_output*2])
est2 = tf.placeholder_with_default(est2_, [None, n_output*2])
seq_len = tf.placeholder_with_default(seq_len_, [None])

# Function for a new thread to load and enqueue data
def load_and_enqueue(sess, enqueue_op, coord, queue_index, total_queues):
    all_training_files = list()
    with open(wav_list_tr, 'r') as f:
        for file,line in enumerate(f):
            line = line.split('\n')[0]
            all_training_files.append(line)

    block_size = len(all_training_files)//total_queues
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
        est1_2batch = list()
        est2_2batch = list()
        time_list = np.zeros([batch_size]).astype('int32')

        for batch_index in range(batch_size):
            y1_y2_x_utt = np.load(base_folder+'/tr/'+all_training_files[current_file+batch_index]+'.npy')
            est1_est2_utt = np.load(base_folder+'/tr/'+all_training_files[current_file+batch_index]+'_simul_est.npy')

            # Load each part of the data
            y1r_2batch.append(np.squeeze(y1_y2_x_utt[0,:,:]))
            y1i_2batch.append(np.squeeze(y1_y2_x_utt[1,:,:]))
            y2r_2batch.append(np.squeeze(y1_y2_x_utt[2,:,:]))
            y2i_2batch.append(np.squeeze(y1_y2_x_utt[3,:,:]))
            xr_2batch.append(np.squeeze(y1_y2_x_utt[4,:,:]))
            xi_2batch.append(np.squeeze(y1_y2_x_utt[5,:,:]))
            time_list[batch_index] =  y1_y2_x_utt.shape[1]

            # Remove utterance-level paddings in STFT estimates
            est1_train_utt = np.squeeze(est1_est2_utt[0,:,:])
            est2_train_utt = np.squeeze(est1_est2_utt[1,:,:])
            input_shape_utt, output_shape_utt, _ = getUnetPadding(np.array([y1_y2_x_utt.shape[1], y1_y2_x_utt.shape[2]]), is_causal=True)
            est1_train_utt = est1_train_utt[(output_shape_utt[0]-y1_y2_x_utt.shape[1]):,:]
            est2_train_utt = est2_train_utt[(output_shape_utt[0]-y1_y2_x_utt.shape[1]):,:] 
            est1_2batch.append(est1_train_utt)
            est2_2batch.append(est2_train_utt)

        # Get longest utterance in a batch
        max_time = np.max(time_list)
        freq_dim = xr_2batch[0].shape[1]
        # Compute input and output shape for all samples in a batch
        input_shape, output_shape, _ = getUnetPadding(np.array([max_time, freq_dim]), is_causal=True)

        # Apply paddings to all training samples
        for batch_index in range(batch_size):
            # Pad all samples in the batch. Pad difference between max_time and output shape (along time axis)
            y1r_2batch[batch_index] = np.pad(y1r_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            y1i_2batch[batch_index] = np.pad(y1i_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            y2r_2batch[batch_index] = np.pad(y2r_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            y2i_2batch[batch_index] = np.pad(y2i_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            xr_2batch[batch_index] = np.pad(xr_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            xi_2batch[batch_index] = np.pad(xi_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            est1_2batch[batch_index] = np.pad(est1_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            est2_2batch[batch_index] = np.pad(est2_2batch[batch_index], [(output_shape[0]-max_time, max_time-time_list[batch_index]), (0,0)], mode='constant', constant_values=0.0)
            # Pad input time (input is larger than ouput)
            padding_frames = (input_shape[0] - output_shape[0]) // 2
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
        est1_2batch = np.vstack(est1_2batch)
        est2_2batch = np.vstack(est2_2batch)

        # enqueue operation
        sess.run(enqueue_op, feed_dict={xr_batch: xr_2batch, xi_batch: xi_2batch, y1r_batch: y1r_2batch, y1i_batch: y1i_2batch, y2r_batch: y2r_2batch, y2i_batch: y2i_2batch, est1_batch: est1_2batch, est2_batch: est2_2batch, seq_len_batch: time_list})

        current_file += batch_size

# Define the assignment w.r.t. PIT cost
def assignment(Y1R,Y1I,Y2R,Y2I,EST1,EST2,_seq_len):
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

    # Form the target label for sequential grouping
    assign = tf.concat([index_11,index_12],axis=1)
    assign = tf.reshape(assign, [-1,2])

    # Calculate complex STFT domain loss (min PIT loss)
    loss = tf.minimum(loss1,loss2)
    loss = tf.reduce_mean(loss)
    loss = loss * tf.reduce_max(tf.to_float(_seq_len)) / tf.reduce_mean(tf.to_float(_seq_len))

    # Calculate frame-level weights for sequential grouping training
    num_seq = tf.shape(_seq_len)[0]
    loss_output = tf.reshape(tf.abs(tf.subtract(loss1,loss2)), [num_seq,-1])
    loss_output_sum = tf.reduce_sum(loss_output,axis=1,keep_dims=True)
    loss_output = tf.div(loss_output,loss_output_sum)
    loss_output = tf.reshape(loss_output, [-1,1])
    W_SEQ = loss_output
    W_SEQ = tf.stop_gradient(W_SEQ)

    return assign, loss, W_SEQ

# Define causal TCN model
def causal_tcn(_est1,_est2,_xor,_xoi,_seq_len,keep_prob):

    with tf.variable_scope("grouping"):
        # Initializer for CNN kernels
        _initializer=tf.truncated_normal_initializer(stddev=0.02)

        # Define input mixture
        num_seq = tf.shape(_seq_len)[0]
        inputR = _xor
        inputR = tf.reshape(inputR, [num_seq, -1, n_input, 1]) # (1, -1, n_input)
        inputR = tf.transpose(inputR, [0,2,1,3])

        inputI = _xoi
        inputI = tf.reshape(inputI, [num_seq, -1, n_input, 1]) # (1, -1, n_input)
        inputI = tf.transpose(inputI, [0,2,1,3])

        # Crop input mixture to the size of Dense-UNet output
        inputR = crop_causal(inputR, crop_diff_list[-1])
        inputI = crop_causal(inputI, crop_diff_list[-1])
        inputR = tf.transpose(inputR, [0,2,1,3])
        inputI = tf.transpose(inputI, [0,2,1,3])

        # Compute magnitude STFT of the mixture
        input_mag = tf.sqrt(tf.square(inputR) + tf.square(inputI))

        # Define input Dense-UNet estimates
        input_est1 = tf.reshape(_est1, [num_seq, -1, n_output, 2])
        input_est2 = tf.reshape(_est2, [num_seq, -1, n_output, 2])

        # Compute magnitude STFT of Dense-UNet estimates
        input_est1_mag = tf.sqrt(tf.reduce_sum(tf.square(input_est1),3,keep_dims=True))
        input_est2_mag = tf.sqrt(tf.reduce_sum(tf.square(input_est2),3,keep_dims=True))

        # Concatenate all information as input to TCN
        X1 = tf.concat([inputR,inputI,input_mag,input_est1,input_est1_mag,input_est2,input_est2_mag],3)

        # A dense block for feature preprocessing
        NUM_INITIAL_FILTERS = 16
        X1_NEW = tf.layers.conv2d(X1, NUM_INITIAL_FILTERS, [1,3], activation=tf.nn.elu, padding='same', kernel_initializer=_initializer)
        X1_NEW = causal_layer_norm(X1_NEW, total_dim=4, name='pre_1')

        X1 = tf.concat([X1, X1_NEW], 3)
        X1_NEW = tf.layers.conv2d(X1, NUM_INITIAL_FILTERS, [1,3], activation=tf.nn.elu, padding='same', kernel_initializer=_initializer)
        X1_NEW = causal_layer_norm(X1_NEW, total_dim=4, name='pre_2')

        X1 = tf.concat([X1, X1_NEW], 3)
        X1_NEW = tf.layers.conv2d(X1, NUM_INITIAL_FILTERS, [1,3], activation=tf.nn.elu, padding='same', kernel_initializer=_initializer)
        X1_NEW = causal_layer_norm(X1_NEW, total_dim=4, name='pre_3')

        X1 = tf.concat([X1, X1_NEW], 3)
        X1_NEW = tf.layers.conv2d(X1, NUM_INITIAL_FILTERS, [1,3], activation=tf.nn.elu, padding='same', kernel_initializer=_initializer)
        X1_NEW = causal_layer_norm(X1_NEW, total_dim=4, name='pre_4')

        # Reshape the output of the dense block
        X1 = tf.reshape(X1_NEW, [num_seq, -1, n_output*NUM_INITIAL_FILTERS])
        # Linearly tranform the preprocessed feature 
        X1 = conv1d(X1, n_bottle, filter_width = 1, stride = 1, stddev=0.02, name = 'input_process')
        X1 = causal_layer_norm(X1, total_dim=3, name='pre_5')

        # The actual TCN
        with tf.variable_scope("TCN"):
            skip_connections = list() # Define skip connections
            for i in range(n_tcn_block):
                for j in range(n_dilation_each_block):
                    dilation = np.power(2.0, j)
                    # Call each residual dilated sub-block
                    X1 = residual_block(X1, rate=dilation, n_dilated_units=n_dilated_units, n_bottle=n_bottle, keep_prob=keep_prob, is_causal=True, scope="res_%d_%d" % (i,dilation))
                skip_connections.append(X1)

            # Add skip connections
            for i in range(n_layers):
                if i != n_layers - 1:
                    gamma_res = tf.get_variable("gamma_res_%d"% i, [1], initializer=tf.constant_initializer(0.0), trainable=True)
                else:
                    gamma_res = tf.get_variable("gamma_res_%d"% i, [1], initializer=tf.constant_initializer(1.0), trainable=True)
                if i == 0:
                    X1_SUM = skip_connections[0] * gamma_res
                else:
                    X1_SUM = X1_SUM + skip_connections[i] * gamma_res

            X1 = X1_SUM

        # Embedding estimation
        with tf.variable_scope("embeddings"):
            embedding = conv1d(X1, embedding_dim, filter_width = 1, stride = 1, stddev=0.02, name = 'embed')
            embedding = tf.tanh(embedding)
            embedding = tf.reshape(embedding, [num_seq, -1, embedding_dim])

    return embedding

# Define the sequential grouping cost
def seq_cost(V, Y, W, _seq_len):
    # V: predicted embedding
    # Y: target label
    # W: weights in objective function
    num_seq = tf.shape(_seq_len)[0]
    Y = tf.reshape(Y, [num_seq, -1, 2])
    W = tf.reshape(W, [num_seq, -1, 1])
    VW = tf.multiply(V, W)
    YW = tf.multiply(Y, W)
    V_batch = tf.matmul(tf.transpose(VW,[0,2,1]), VW)
    VY_batch = tf.matmul(tf.transpose(VW,[0,2,1]), YW)
    YV_batch = tf.matmul(tf.transpose(VW,[0,2,1]), YW)
    Y_batch = tf.matmul(tf.transpose(YW,[0,2,1]), YW)

    cost_per_sample = tf.reduce_sum(tf.square(V_batch),[1,2]) - tf.reduce_sum(tf.square(VY_batch),[1,2]) - tf.reduce_sum(tf.square(YV_batch),[1,2]) + tf.reduce_sum(tf.square(Y_batch),[1,2])
    mean_cost = 10000*tf.reduce_mean(cost_per_sample) 

    return mean_cost

# Definition of the network and loss
assign, cost_pit, w_seq = assignment(y1r, y1i, y2r, y2i, est1, est2, seq_len)
embed = causal_tcn(est1, est2, xr, xi, seq_len, keep_prob)
cost = seq_cost(embed, assign, w_seq,  seq_len)

# Define loss and optimizer
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    if FLAGS.is_adam == 1:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta2=0.9).minimize(cost) # Adam Optimizer
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(cost) # Momentum Optimizer

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
        num_threads = 5 # Use 5 threads for data loading, each thread touches different part of training data
        t = [threading.Thread(target=load_and_enqueue, args=(sess,enqueue_op,coord,i,num_threads))  for i in range(num_threads)]
        for tmp_t in t:
            tmp_t.start()
    
    # Use pickle file to save temporary training specs
    training_info_pickle = model_dir+'/training_info.pickle'
    if os.path.isfile(training_info_pickle):
        # If there is a pickle file, then load to restore previous training states
        with open(training_info_pickle, 'rb') as f_pickle:
            step, best_cv_cost, best_cv_step, learning_rate = pickle.load(f_pickle)
        saver.restore(sess, os.getcwd()+"/exp/"+FLAGS.exp_name+"/models/"+FLAGS.time_stamp+'/'+FLAGS.exp_name+"_model.ckpt" + "_step_"+ str(step))
    else:
        # If no pickle file, then initialize training specs
        step = 1
        best_cv_cost = 1000
        best_cv_step = 0

    # Training cost and snr
    train_cost_sum = 0
    train_cost_pit_sum = 0
    total_num = 0
    # CV data reader
    cv_buffer = list()
    cv_est_buffer = list()
    cv_loaded = 0

    # Keep training until reaching max iterations
    while step  < FLAGS.epoch * total_tr_files / batch_size:
        # At the beginning of each epoch
        # If FLAGS.is_deploy is 0, run validation at the beginning of each epoch
        # IF FLAGS.is_deploy is 1, run evaluation and exit
        if step % (total_tr_files // batch_size) == 1: 
            
            if FLAGS.is_deploy == 0:
                # CV 
                logging.info('Start to CV')
                cost_list = list() 
                cost_list_pit = list()
                with open(wav_list_cv, 'r') as f:
                    for file,line in enumerate(f):
                        if file>=500: # TODO If want to skip some cv files
                            break
                        line = line.split('\n')[0]
                        # Load cv files
                        if cv_loaded == 0:
                            # If not loaded, load from file system
                            y1_y2_x_utt = np.load(base_folder+'/cv/'+line+'.npy')
                            est1_est2_utt = np.load(base_folder+'/cv/'+line+'_simul_est.npy')
                            cv_buffer.append(y1_y2_x_utt)
                            cv_est_buffer.append(est1_est2_utt)
                        else:
                            # If loaded, load from buffer
                            y1_y2_x_utt = cv_buffer[file]
                            est1_est2_utt = cv_est_buffer[file]

                        # load features and targets
                        y1r_utt, y1i_utt, y2r_utt, y2i_utt, xr_utt, xi_utt, sig_utt, utt_len, time_tmp, \
                            y1r_utt_pad, y1i_utt_pad, y2r_utt_pad, y2i_utt_pad, xr_utt_pad, xi_utt_pad, \
                            sig_utt_pad = single_utt_feature_proc(y1_y2_x_utt, None, is_causal=True)

                        est_1 = np.squeeze(est1_est2_utt[0,:,:])
                        est_2 = np.squeeze(est1_est2_utt[1,:,:])

                        test_cost, test_cost_pit = sess.run([cost, cost_pit], \
                        feed_dict={xr: xr_utt_pad, xi: xi_utt_pad, y1r: y1r_utt_pad, y1i: y1i_utt_pad, y2r: y2r_utt_pad, \
                            y2i: y2i_utt_pad, est1: est_1, est2: est_2, seq_len: utt_len, keep_prob: 1, is_training: False})
                        cost_list.append(test_cost)
                        cost_list_pit.append(test_cost_pit)

                # Set flag to 1
                cv_loaded = 1

                # Calculate mean loss and snr
                mean_cost = np.mean(cost_list[:])
                mean_cost_pit = np.mean(cost_list_pit[:])
                
                # If best cv cost, update the corresponding variable
                if mean_cost <= best_cv_cost:
                    best_cv_cost = mean_cost
                    best_cv_step = step
                    logging.info('Best cv cost !!!!!!!!!!!!!!!!!!!') 
                
                # Output cv loss
                logging.info('cv_cost %f',mean_cost)
                logging.info('cv_cost_pit %f',mean_cost_pit)
            else:
                # Evaluation
                with open(wav_list_tt, 'r') as f:
                    for file,line in enumerate(f):
                        line = line.split('\n')[0]
                        y1_y2_x_utt = np.load(base_folder+'/tt/'+line+'.npy')
                        est1_est2_utt = np.load(base_folder+'/tt/'+line+'_simul_est.npy')

                        # load features and targets
                        y1r_utt, y1i_utt, y2r_utt, y2i_utt, xr_utt, xi_utt, sig_utt, utt_len, time_tmp, \
                            y1r_utt_pad, y1i_utt_pad, y2r_utt_pad, y2i_utt_pad, xr_utt_pad, xi_utt_pad, \
                            sig_utt_pad = single_utt_feature_proc(y1_y2_x_utt, None, is_causal=True)

                        est_1 = np.squeeze(est1_est2_utt[0,:,:])
                        est_2 = np.squeeze(est1_est2_utt[1,:,:])

                        start_time = time.time()
                        test_cost, embedding_test=sess.run([cost,embed],\
                            feed_dict={xr: xr_utt_pad, xi: xi_utt_pad, y1r: y1r_utt_pad, y1i: y1i_utt_pad, y2r: y2r_utt_pad, \
                                y2i: y2i_utt_pad, est1: est_1, est2: est_2, seq_len: utt_len, keep_prob: 1, is_training: False})
                        elapsed_time = time.time() - start_time
                        logging.info(line +' cost:' + str(test_cost) + ' time:' + str(elapsed_time))

                        # Reshape and crop spectral estimates
                        input_shape_test, output_shape_test, _ = getUnetPadding(np.array([xr_utt.shape[0], xr_utt.shape[1]]), n_layers=4, is_causal=True)
                        est_1 = np.reshape(est_1, [-1, output_shape_test[1], 2])
                        est_2 = np.reshape(est_2, [-1, output_shape_test[1], 2])
                        est_1 = est_1[(output_shape_test[0]-time_tmp):,:-1,:]
                        est_2 = est_2[(output_shape_test[0]-time_tmp):,:-1,:]
                        est_1 = np.reshape(est_1, (-1, 2*feat_dim))
                        est_2 = np.reshape(est_2, (-1, 2*feat_dim))

                        # Reshape and crop embedding estimates
                        embedding_test = np.squeeze(embedding_test)
                        embedding_test = embedding_test[(output_shape_test[0]-time_tmp):,:]

                        '''
                        # Non-causal clustering
                        # Get frames with significant energy
                        energy_est1 = np.sum(np.square(est_1),1)
                        w1_est = np.reshape((energy_est1>=2e-3*np.max(energy_est1)),[-1,1])
                        energy_est2 = np.sum(np.square(est_2),1)
                        w2_est = np.reshape((energy_est2>=2e-3*np.max(energy_est2)),[-1,1])
                        w_est = np.concatenate((w1_est,w2_est),1)
                        w_est = np.amax(w_est,axis=1)
                        w_est_flat = np.reshape(w_est,[-1])

                        # K-means for frames with significant energy
                        embedding_test_with_energy = embedding_test[w_est_flat,:]
                        clu = cluster.k_means(embedding_test_with_energy.astype('float64'),2)
                        centroids  = clu[0].astype('float32')
                        
                        # Iterate all frames, get estimated assign
                        embedding_test_2d = np.reshape(embedding_test,[-1,embedding_dim])
                        assigns = np.ones(w_est.shape) *(-1)
                        for time_frame in range(w_est.shape[0]):
                            embed_1 = np.reshape(embedding_test_2d[time_frame,:],[1,-1])
                            assigns[time_frame] = np.argmin(-np.sum((centroids*embed_1),1))
                        '''

                        # Start 2-speaker causal clustering
                        embedding_test_2d = np.reshape(embedding_test,[-1,embedding_dim])
                        # Similarity of the neighboring two frames
                        embedding_test_similarity = np.sum((embedding_test_2d[:-1,:] * embedding_test_2d[1:,:]), 1, keepdims=True)
                        embedding_test_similarity = np.concatenate((np.zeros([1,1]), embedding_test_similarity), 0)
                        # If the neighboring has similarity >= 0.5, then they are similar
                        neighboring_embedding_is_similar = np.asarray(embedding_test_similarity>=0.5, 'float32')

                        # Frame-level energy
                        frame_energy = np.sum(np.square(xr_utt)+np.square(xi_utt), 1)
                        # Maximum energy
                        max_energy = 0

                        # Indicator of empty queue
                        embedding_queue1_empty = True
                        embedding_queue2_empty = True

                        # Iterate all frames, get estimated assign
                        assigns = np.ones(embedding_test_similarity.shape) *(-1)
                        for time_frame in range(embedding_test_similarity.shape[0]):
                            embed_t = np.reshape(embedding_test_2d[time_frame,:],[1,-1])
                            max_energy = np.maximum(frame_energy[time_frame], max_energy)
                            if time_frame == 0: 
                                # First frame, intitalize queue 1
                                embedding_queue1_empty = False
                                embedding_queue1 = embed_t
                                assigns[time_frame] = 0
                            else:
                                if embedding_queue2_empty:
                                    # No item in queue 2
                                    if neighboring_embedding_is_similar[time_frame] == 1:
                                        assigns[time_frame] = 0
                                        if frame_energy[time_frame] > 0.3 * max_energy:
                                            embedding_queue1 = np.concatenate((embedding_queue1, embed_t), 0)
                                    elif neighboring_embedding_is_similar[time_frame] == 0:
                                        assigns[time_frame] = 1
                                        embedding_queue2 = embed_t
                                        embedding_queue2_empty = False
                                else:
                                    # Queue 2 is not empty
                                    if np.sum((embed_t * embedding_queue1_mean)) > np.sum((embed_t * embedding_queue2_mean)):
                                        assigns[time_frame] = 0
                                        if frame_energy[time_frame] > 0.3 * max_energy:
                                            embedding_queue1 = np.concatenate((embedding_queue1, embed_t), 0)
                                    else:
                                        assigns[time_frame] = 1
                                        if frame_energy[time_frame] > 0.3 * max_energy:
                                            embedding_queue2 = np.concatenate((embedding_queue2, embed_t), 0)
                            if not embedding_queue1_empty:
                                if  embedding_queue1.shape[0] > 10:
                                    # Remove items from the queue if oversize
                                    embedding_queue1 = embedding_queue1[1:,:]
                                embedding_queue1_mean = np.mean(embedding_queue1, 0, keepdims=True)
                            if not embedding_queue2_empty:
                                if  embedding_queue2.shape[0] > 10:
                                    # Remove items from the queue if oversize
                                    embedding_queue2 = embedding_queue2[1:,:]
                                embedding_queue2_mean = np.mean(embedding_queue2, 0, keepdims=True)

                        # Reassign estimates according to the estimated labels
                        est_all = np.concatenate((np.expand_dims(est_1,-1),np.expand_dims(est_2,-1)),2)
                        assigns = np.reshape(assigns, (-1,1,1))
                        assigns_mult = np.concatenate((assigns*1.0, 1.0-assigns),2)
                        est_1 = np.sum(assigns_mult*est_all, 2)
                        est_2 = np.sum((1.0-assigns_mult)*est_all, 2)

                        est_1 = np.reshape(est_1, (-1, feat_dim, 2))
                        est_2 = np.reshape(est_2, (-1, feat_dim, 2))
                        est_1r = np.squeeze(est_1[:,:,0])
                        est_1i = np.squeeze(est_1[:,:,1])
                        est_2r = np.squeeze(est_2[:,:,0])
                        est_2i = np.squeeze(est_2[:,:,1])

                        # load original wav file
                        wav_folders = data_folder + '/2speakers/wav8k/min/tt/'
                        sr,clean_audio_1 = wav_read(wav_folders+'s1/'+line+'.wav')
                        clean_audio_1 = clean_audio_1.astype('float32')/np.power(2,15)
                        sr,clean_audio_2 = wav_read(wav_folders+'s2/'+line+'.wav') 
                        clean_audio_2 = clean_audio_2.astype('float32')/np.power(2,15)
                        sr,mix_audio = wav_read(wav_folders+'mix/'+line+'.wav')
                        mix_audio = mix_audio.astype('float32')/np.power(2,15)

                        # Compute time-domain estimated signals
                        RES_1 = est_1r + 1j* est_1i
                        RES_2 = est_2r + 1j* est_2i
                        RES_1 = np.concatenate((RES_1, np.conj(RES_1[:,::-1][:,1:-1])), axis=1)
                        RES_2 = np.concatenate((RES_2, np.conj(RES_2[:,::-1][:,1:-1])), axis=1)
                        res_1 = istft(RES_1,len(clean_audio_1))
                        res_2 = istft(RES_2,len(clean_audio_2))
                        res_1 = res_1.astype('float32')
                        res_2 = res_2.astype('float32')

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
            with open(training_info_pickle, 'wb') as f_pickle:
                pickle.dump([step, best_cv_cost, best_cv_step, learning_rate], f_pickle)
            logging.info('Model saved in file: %s',save_path)
        
        # Apply learning rate decay
        # Check every three epochs, if the cv snr does not increase for more than 4 epochs, apply lr_decay
        if (step % (3*(total_tr_files // batch_size)) == 1) and (step != 1):
            if step - best_cv_step >= int(3.9*(total_tr_files / batch_size)):
                learning_rate *= FLAGS.lr_decay

        # Run one batch of training
        _, train_cost, train_cost_pit, q_size\
        = sess.run([optimizer, cost, cost_pit, queue_size], \
        feed_dict={lr: learning_rate, keep_prob: FLAGS.keep_prob, is_training: True})

        # Update training loss
        train_cost_sum += train_cost
        train_cost_pit_sum += train_cost_pit
        total_num += 1

        # If reaches display step, print loss
        if step % display_step == 1:
            logging.info('Step %d/%d, Minibatch Loss=%f, Minibatch Loss PIT=%f, Q_size=%d, Learning Rate=%f',step, FLAGS.epoch * total_tr_files / batch_size, train_cost_sum/total_num, train_cost_pit_sum/total_num, q_size, learning_rate)
            # Reset all loss variables
            train_cost_sum = 0
            train_cost_pit_sum = 0
            total_num = 0
        
        step += 1

    sys.exit(0)
