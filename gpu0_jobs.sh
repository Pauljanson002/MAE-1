#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
echo 'Starting job: scheduler=cosine, reduction_factor=0, method=lwf, cooldown_ratio=0, constant_lr_ratio=0, alpha=0.5, lamda=0'
bin/batch.sh "cosine" "0" "lwf" "0" "exp_6" "0" "0" "0.5" "0"
sleep 1
