#!/bin/bash
#Called by ml_condor/condor_job.sh inside condor container

#This script executes the neural network training on the condor cluster using GPU
#1. The needed datasets are copied to a directory accesable to the container (/tmp/)
#2. All used programms are sourced. Tensorflow is sourced with a different version from LCG_95 (tensorflow-gpu=1.12.0)
#3. The training is started
#4. All produced .h5, .png, .pdf and .log files are copied to the directory that the cluster sends back

# Steps 2 and 3 are functionally identical to ml/run_training.sh

set -e
ERA_NAME=$1
CHANNEL=$2
TAG=$3
NAME_USER=$4
#---1---
cephdir=root://ceph-node-a.etp.kit.edu:1094//${NAME_USER}/nmssm_data
if [[ $ERA_NAME == "all_eras" ]]; then
  ERAS="2016 2017 2018"
else
 ERAS=$ERA
fi

echo $ERAS

# Copy the needed datasets from /ceph
for ERA in ${ERAS}; do
  folder=${ERA}_${CHANNEL}_${TAG}
  outdir=/tmp/${folder}
  mkdir -p ${outdir_loop}
  xrdcp -r ${cephdir}/${folder}/fold0_training_dataset.root ${cephdir}/${folder}/fold1_training_dataset.root ${outdir}
  echo copy ${folder}
done
# Unpacks htt-ml and utils directories
tar -xf httml.tar.gz 
#Fix paths of config-file to apply in container (output/ml/->/tmp/)
sed -e 's@output/ml@/tmp@g' -i dataset_config.yaml
#---2---
source utils/bashFunctionCollection.sh
export KERAS_BACKEND=tensorflow
export OMP_NUM_THREADS=12
export THEANO_FLAGS=gcc.cxxflags=-march=corei7
source utils/setup_cvmfs_sft.sh
export PYTHONUSERBASE=$HOME/.local/pylibs-$(hostname)
export PYTHONPATH=/usr/local/python2.7/site-packages:$HOME/.local/pylibs-$(hostname)/lib/python2.7/site-packages:$PYTHONPATH
export PATH=/usr/local:/usr/local/cuda-9.0/bin:$HOME/.local/pylibs-$(hostname)/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

#---3---
if [[ $ERA == "all_eras" ]]
then
  mkdir -p $outdir
  python htt-ml/training/keras_training.py dataset_config.yaml 0 --balance-batches 1 --conditional 1 #--randomization 1
  python htt-ml/training/keras_training.py dataset_config.yaml 1 --balance-batches 1 --conditional 1 #--randomization 1
else
  mkdir -p $outdir
  python htt-ml/training/keras_training.py dataset_config.yaml 0 --balance-batches 1
  python htt-ml/training/keras_training.py dataset_config.yaml 1 --balance-batches 1
fi
#---4---
folder=${ERA_NAME}_${CHANNEL}_${TAG}
outdir=/tmp/${folder}
mkdir condor_output_${folder}
cp ${outdir}/*.h5 ${outdir}/*.png ${outdir}/*.pdf ${outdir}/*.log ${outdir}/*.pickle condor_output_${folder}
