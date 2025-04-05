#!/bin/bash

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4
module load mpi4py/3.1.4

# Create output directory if it doesn't exist
mkdir -p output

# Function to run the sentiment analysis
run_sentiment_analysis() {
    local nodes=$1
    local cores=$2
    local config_name=$3
    
    echo "Running with $nodes node(s) and $cores core(s)..."
    
    if [ "$nodes" -eq 2 ]; then
        # Using srun for 2 nodes (with 4 cores per node)
        srun -n 8 --ntasks-per-node=4 --nodes=2 python sentiment_analysis.py mastodon-16m.ndjson --batch-size 5000
    else
        # Using srun for 1 node with the specified number of cores
        srun -n $cores python sentiment_analysis.py mastodon-16m.ndjson --batch-size 5000
    fi
}

# Submit job with 1 node, 1 core
echo "Submitting job with 1 node, 1 core..."
JOB1=$(sbatch --nodes=1 --ntasks=1 \
    --job-name=sentiment_1n1c \
    --output=output/sentiment_1n1c_%j.out \
    --error=output/sentiment_1n1c_%j.err \
    --time=6:00:00 \
    --wrap="$(declare -f run_sentiment_analysis); run_sentiment_analysis 1 1 '1n1c'")

echo "Submitted job 1: ${JOB1}"

# Submit job with 1 node, 8 cores
echo "Submitting job with 1 node, 8 cores..."
JOB2=$(sbatch --nodes=1 --ntasks=8 \
    --job-name=sentiment_1n8c \
    --output=output/sentiment_1n8c_%j.out \
    --error=output/sentiment_1n8c_%j.err \
    --time=4:00:00 \
    --wrap="$(declare -f run_sentiment_analysis); run_sentiment_analysis 1 8 '1n8c'")

echo "Submitted job 2: ${JOB2}"

# Submit job with 2 nodes, 8 cores (4 per node)
echo "Submitting job with 2 nodes, 8 cores (4 per node)..."
JOB3=$(sbatch --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    --job-name=sentiment_2n8c \
    --output=output/sentiment_2n8c_%j.out \
    --error=output/sentiment_2n8c_%j.err \
    --time=5:00:00 \
    --wrap="$(declare -f run_sentiment_analysis); run_sentiment_analysis 2 8 '2n8c'")

echo "Submitted job 3: ${JOB3}"

echo "All jobs submitted!"
