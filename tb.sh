#!/bin/bash
cd ~/spinnup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pyrep_spinningup
tensorboard --logdir output
