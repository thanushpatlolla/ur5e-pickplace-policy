#!/bin/bash

echo "Starting 3 training runs with different hyperparameters..."

echo ""
echo "========================================="
echo "Run 1: Gentler scheduler"
echo "========================================="
python train.py --lr_scheduler_factor 0.5 --lr_scheduler_patience 5

echo ""
echo "========================================="
echo "Run 2: more weight decay"
echo "========================================="
python train.py --weight_decay 3e-4

echo ""
echo "========================================="
echo "Run 3: both"
echo "========================================="
python train.py --lr_scheduler_factor 0.5 --lr_scheduler_patience 5 --weight_decay 3e-4

echo ""
echo "========================================="
echo "All runs completed!"
echo "========================================="
