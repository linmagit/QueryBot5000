#!/bin/bash

# Copy and decompress the sample data file
fileid="1imVPNXk8mGU0v9OOhdp0d9wFDuYqARwZ"
filename="tiramisu-sample.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
tar -xvzf ${filename}

# Generate and combine query templates
./pre-processor/templatizer.py tiramisu --dir tiramisu-sample/ --output templates
./pre-processor/csv-combiner.py --input_dir templates/ --output_dir tiramisu-combined-csv

# Run through clustering algorithm
./clusterer/online_clustering.py --dir tiramisu-combined-csv/ --rho 0.8
./clusterer/generate-cluster-coverage.py --project tiramisu --assignment online-clustering-results/None-0.8-assignments.pickle --output_csv_dir online-clusters/ --output_dir cluster-coverage/

# Run forecasting models
./forecaster/run_sample.sh

# Generate ENSEMBLE and HYBRID results
./forecaster/generate_ensemble_hybrid.py prediction-results/agg-60/horizon-4320/ar/ prediction-results/agg-60/horizon-4320/noencoder-rnn/ prediction-results/agg-60/horizon-4320/ensemble False
./forecaster/generate_ensemble_hybrid.py  prediction-results/agg-60/horizon-4320/ensemble prediction-results/agg-60/horizon-4320/kr prediction-results/agg-60/horizon-4320/hybrid True
