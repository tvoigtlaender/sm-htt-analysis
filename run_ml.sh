#!/bin/bash

ERA=$1
CHANNEL=$2
LOSS=$3

./ml/create_training_dataset.sh $ERA $CHANNEL
#./ml/sum_training_weights.sh $ERA
./ml/run_training.sh $ERA $CHANNEL $LOSS
./ml/run_testing.sh $ERA $CHANNEL
./ml/run_application.sh $ERA $CHANNEL
