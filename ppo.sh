#!/bin/bash
cd ~/spinnup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pyrep_spinningup
if [ ! $1 ] || [ $1 == 'train' ]
then
  
  if [ ! $2 ]
  then
    echo '    ----  Training PPO  ----    '
    python spinup/algos/pytorch/ppo/ppo.py --headless
  fi

  if [ $2 == 'render' ]
  then
    echo '    ----  Training PPO  ----    '
    python spinup/algos/pytorch/ppo/ppo.py
  fi

fi
