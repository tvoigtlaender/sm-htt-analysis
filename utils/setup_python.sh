#!/bin/bash

# Add all python scripts/modules of sm-htt-analysis to PYTHONPATH
STARTING_PATH=$(realpath $(dirname $(dirname ${BASH_SOURCE[0]})))
export PYTHONPATH=$PYTHONPATH:$STARTING_PATH/shape-producer
export PYTHONPATH=$PYTHONPATH:$STARTING_PATH/datacard-producer
export PYTHONPATH=$PYTHONPATH:$STARTING_PATH/Dumbledraw
export PYTHONPATH=$PYTHONPATH:$STARTING_PATH/tensorflow_derivative
export PYTHONPATH=$PYTHONPATH:$STARTING_PATH/utils
export PYTHONPATH=$PYTHONPATH:$STARTING_PATH/fake-factor-application
export PYTHONPATH=$PYTHONPATH:$STARTING_PATH/jdl_creator
