#!/bin/bash

# Script to test all best models in checkpoint subdirectories
# Runs headless and saves output for each run

# Create results directory if it doesn't exist
RESULTS_DIR="test_results"
mkdir -p "$RESULTS_DIR"

# Find all subdirectories in checkpoints/
for checkpoint_dir in checkpoints/*/; do
    # Get the directory name (e.g., 20260101_032252)
    dir_name=$(basename "$checkpoint_dir")

    # Check if best_model.pth exists in this directory
    if [ -f "${checkpoint_dir}best_model.pth" ]; then
        echo "======================================"
        echo "Testing model: $dir_name"
        echo "======================================"

        # Output file for this run
        output_file="${RESULTS_DIR}/${dir_name}_results.txt"
        json_file="${RESULTS_DIR}/${dir_name}_results.json"

        # Run test_model.py headless and save output
        python test_model.py \
            --checkpoint "${checkpoint_dir}best_model.pth" \
            --num-runs 100 \
            --no-viewer \
            --save-results "$json_file" \
            2>&1 | tee "$output_file"

        echo "Results saved to: $output_file"
        echo ""
    else
        echo "Warning: No best_model.pth found in $checkpoint_dir"
    fi
done

echo "======================================"
echo "All tests complete!"
echo "Results saved in: $RESULTS_DIR/"
echo "======================================"
