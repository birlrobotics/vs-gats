#! /usr/bin/env bash

#echo -n 'Before using this script, have you reset the res_dir variable in hico_constants.py file?(y/n)' && read x 
# FEAT_TYPE=$1
EXP_VER=$1
CHECKPOINT=$2
read -p 'Before using this script, have you reset the res_dir variable in hico_constants.py file?(y/n)' x
if [[ $x='y' ]]; then
    echo 'running eval.py file to get prediction result '
    python -m eval --e_v=$EXP_VER -p=$CHECKPOINT # --f_t=$FEAT_TYPE
    echo 'running result/compute_map.py to compute map'
    python -m result.compute_map \
        --e_v=$EXP_VER
    echo 'running result/sample_analysis.py to get splited map'
    python -m result.sample_analysis --e_v=$EXP_VER
else
    echo 'Please set the variable first'
fi