# Deep CASA for talker-independent monaural speaker separation

## Introduction

This is Tensorflow implementation of "Divide and conquer: A deep CASA approach to talker-independent monaural speaker separation", IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 27, pp. 2092-2102. 

## Contents

* `./feat/exp_prepare_folder.sh`: prepares folders for experiments.
* `./feat/feat_gen.py`: generates STFT featuers for training, validation and test.
* `./feat/stft.py`: defines STFT and iSTFT.
* `./nn/simul_group.py`: training/validation/test of the simultaneous grouping stage.
* `./nn/seq_group.py`: training/validation/test of the sequential grouping stage.
* `./nn/utility.py`: defines various functions in `simul_group.py` and `seq_group.py`.

## Experimental setup

This codebase has been tested on AWS EC2 p3.2xlarge nodes with Deep Learning AMI (Ubuntu 16.04) Version 26.0.

Follow instructions in turn to set up the environment and run experiments.

1. Requirements:
    * Tensorflow 1.15.0. <br />
        Activate the environment on EC2:
        ```
        source activate tensorflow_p27
        ```
    * gflags
        ```
        pip install python-gflags
        ```
    * Please install other necessary python packages if not using AWS deep Learning AMI (Ubuntu 16.04) Version 26.0.
    
2. Before running experiments, activate the tensorflow environment on EC2 using:
    ```
    source activate tensorflow_p27
    ```

3. Generate the WSJ0-2mix dataset using `http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip`. Copy the generated files to the EC2 instance.

4. Start feature extraction by running the following command in the main directory:
    ```
    python feat/feat_gen.py
    ```
    Thre are two arguments in `feat_gen.py`, `data_folder` and `wav_list_folder`. Change them to where your WSJ0-2mix dataset and file list locate. 

5. Train the simultaneous grouping stage using:
    ```
    TIME_STAMP=train_simul_group
    python nn/simul_group.py --time_stamp $TIME_STAMP --is_deploy 0 --batch_size 1 
    ```
    * Due to utterance-level training and limited GPU memory, `batch_size` can be selected as 1 or 2. 
    * Change `data_folder` and `wav_list_folder` accordingly. 
    * You can also change other hyperparameters, e.g., the number of epochs and learning rate, using gflags arguments.
   
6. Run inference of simultaneous grouping (tt set) using:
    ```
    RESUME_MODEL=exp/deep_casa_wsj/models/train_simul_group/deep_casa_wsj_model.ckpt_step_1
    python nn/simul_group.py --is_deploy 1 --resume_model $RESUME_MODEL
    ```
    * `$RESUME_MODEL` is the model to be loaded for inference. Change it accordingly.
    * Mixtures, clean references and Dense-UNet estimates will be generated and saved in folder `./exp/deep_casa_wsj/output_tt/files/`. 
    * Please use your own scripts to generate results in different metrics.

7. Generate temporary .npy file for the next stage (sequential grouping):
    ```
    RESUME_MODEL=exp/deep_casa_wsj/models/train_simul_group/deep_casa_wsj_model.ckpt_step_1
    python nn/simul_group.py --is_deploy 2 --resume_model $RESUME_MODEL
    ```
    * Setting `is_deploy` to 2 will generate unorganized estimates by Dense-UNet, and save them as .npy files for the sequential grouping stage. 
    * tr, cv and tt data are generated in turn, and saved in `./exp/deep_casa_wsj/feat/`.

8. Train the sequential grouping stage using:
    ```
    TIME_STAMP=train_seq_group
    python nn/seq_group.py --time_stamp $TIME_STAMP --is_deploy 0
    ```
    Change `data_folder` and `wav_list_folder` accordingly. You can also change other hyperparameters, e.g., the number of epochs and learning rate, using gflags arguments.
    
9. Run inference of sequential grouping (tt set) using:
   ```
    RESUME_MODEL=exp/deep_casa_wsj/models/train_seq_group/deep_casa_wsj_model.ckpt_step_1
    python nn/seq_group.py --is_deploy 1 --resume_model $RESUME_MODEL
   ```
    * `$RESUME_MODEL` is the model to be loaded for inference. Change it accordingly.
    * Mixtures, clean references and estimates will be saved in folder `./exp/deep_casa_wsj/output_tt/files/`.
    * Please use your own scripts to generate results in different metrics. 
