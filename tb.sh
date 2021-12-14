#!/bin/bash
cd ~/spinnup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate spinningup
tensorboard --logdir output
