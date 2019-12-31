import numpy as np
import gflags, os, sys, subprocess
from scipy.io.wavfile import read as wav_read
from stft import stft,istft

gflags.DEFINE_string('data_folder','/home/ubuntu/data/wsj0_2mix','Path to wsj0-2mix data set')
gflags.DEFINE_string('wav_list_folder','/home/ubuntu/data/wsj0_2mix','Folder that stores wsj0-2mix wav list')
FLAGS = gflags.FLAGS
FLAGS(sys.argv)

# Define folders
base_folder= os.getcwd()
data_folder = FLAGS.data_folder
wav_list_folder = FLAGS.wav_list_folder

# Create experiment folder
exp_name='deep_casa_wsj' # the experiment name
exp_folder=base_folder +'/exp/'+ exp_name #the path for the experiment
subprocess.call(base_folder + '/feat/exp_prepare_folder.sh '+ exp_name, shell=True)

wav_list_prefix = wav_list_folder + '/mix_2_spk_min'
wav_path = data_folder + '/2speakers/wav8k/min/'
feat_path = exp_folder+'/feat' #feature path

# Generate feature and save as .npy
def get_feat(wav_list_prefix, wav_path, feat_path, task, fftsize=256, hopsize=64):
    wav_folders = wav_path + task + '/'
    wav_list = wav_list_prefix + '_' +task +'_mix'
    output_dir = feat_path + '/' + task + '/'
    with open(wav_list, 'r') as f:
        for file,line in enumerate(f):
            print(task + ' file: ' + str(file+1))
            # Load wav files
            line = line.split('\n')[0]
            sr,clean_audio_1 = wav_read(wav_folders+'s1/'+line+'.wav')
            clean_audio_1 = clean_audio_1.astype('float32')/np.power(2,15)
            sr,clean_audio_2 = wav_read(wav_folders+'s2/'+line+'.wav')
            clean_audio_2 = clean_audio_2.astype('float32')/np.power(2,15)
            sr,mix_audio = wav_read(wav_folders+'mix/'+line+'.wav')        
            mix_audio = mix_audio.astype('float32')/np.power(2,15)
            # STFT
            Zxx_1 = stft(clean_audio_1)
            Zxx_2 = stft(clean_audio_2)
            Zxx_mix = stft(mix_audio)
            Zxx_1 = Zxx_1[:,0:(fftsize/2+1)]
            Zxx_2 = Zxx_2[:,0:(fftsize/2+1)]
            Zxx_mix = Zxx_mix[:,0:(fftsize/2+1)]
            # Store real and imaginary STFT of speaker1, speaker2 and mixture
            Zxx = np.stack((np.real(Zxx_1).astype('float32'),np.imag(Zxx_1).astype('float32'),np.real(Zxx_2).astype('float32'),np.imag(Zxx_2).astype('float32'),np.real(Zxx_mix).astype('float32'),np.imag(Zxx_mix).astype('float32')),axis=0)
            # Save features and targets to npy files
            np.save(output_dir+line, Zxx)
            # Save time-domain waveform to npy file
            audio_len = range(0, len(clean_audio_1)-fftsize+1, hopsize)[-1] + fftsize 
            audio = np.stack((clean_audio_1[:audio_len], clean_audio_2[:audio_len], mix_audio[:audio_len]), axis=0)
            np.save(output_dir+line+'_wave', audio)
            
# Feature generation for training, cv and test
get_feat(wav_list_prefix, wav_path, feat_path, 'tr', fftsize=256, hopsize=64)
get_feat(wav_list_prefix, wav_path, feat_path, 'cv', fftsize=256, hopsize=64)
get_feat(wav_list_prefix, wav_path, feat_path, 'tt', fftsize=256, hopsize=64)
