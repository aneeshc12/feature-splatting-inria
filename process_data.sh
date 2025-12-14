#!/bin/bash

set -e

seq=$1
op=$1_output
python convert.py -s $seq
python compute_obj_part_feature.py -s $seq

python train.py -s $seq -m $op --iterations 30000