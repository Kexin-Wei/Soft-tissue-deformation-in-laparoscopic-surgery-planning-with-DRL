#!/bin/bash
cd ~/spinnup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate spinningup
if [ ! $1 ] || [ $1 == 'train' ]
then
  
  if [ ! $2 ]
  then
    echo '    ----  Training DDPG  ----    '
    python spinup/algos/pytorch/ddpg/ddpg.py --headless
  fi

  if [ $2 == 'render' ]
  then
    echo '    ----  Training DDPG  ----    '
    python spinup/algos/pytorch/ddpg/ddpg.py
  fi

fi
