#!/bin/bash

echo "Starting 3 training runs with different hyperparameters..."

echo ""
echo "========================================="
echo "Run 1: same batch, larger lr"
echo "========================================="
python train.py --lr 5e-4

echo ""
echo "========================================="
echo "Run 2: smaller batch"
echo "========================================="
python train.py --batch_size 128

echo ""
echo "========================================="
echo "Run 3: smaller batch, larger lr"
echo "========================================="
python train.py --lr 5e-4 --batch_size 128

echo ""
echo "========================================="
echo "Run 4: smaller batch, less weight decay"
echo "========================================="
python train.py --batch_size 128 --weight_decay 1e-4

echo ""
echo "========================================="
echo "Run 5: smaller batch, less weight decay, larger lr"
echo "========================================="
python train.py --batch_size 128 --weight_decay 2e-4 --lr 5e-4

echo ""
echo "========================================="
echo "All runs completed!"
echo "========================================="
