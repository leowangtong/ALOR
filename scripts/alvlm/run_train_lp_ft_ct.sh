#!/bin/bash

# methods=("probing" "finetune" "FLYP")
methods=("FLYP")

folder="retrieved_al"

# cls_inits=("random" "text" "REAL-Prompt" )
cls_inits=("text")

batch_size=32

epochs=50
#epochs=1 # for quick testing only

model_cfg="vitb32_openclip_laion400m"

# log_mode="file"
log_mode="both"


# Check if command-line arguments were provided
if [ "$#" -ge 1 ]; then
    datasets=("$1")  # Use the provided command-line argument for the dataset
else
    echo "Usage: $0 <dataset> [seed]"
fi

if [ "$#" -ge 2 ]; then
    seeds=("$2")  # Use the provided command-line argument for the seed
    
fi

if [ "$#" -ge 3 ]; then
    ALMETHOD=("$3")  # Use the provided command-line argument for the seed
fi

if [ "$#" -ge 4 ]; then
    rounds=("$4")  # Use the provided command-line argument for the seed
fi

# Check if the results folder exists, if not create it
if [ ! -d "results/$folder" ]; then
    mkdir -p "results/$folder"
fi

output_folder="output/$folder"
if [ ! -d "$output_folder" ]; then
    mkdir -p "$output_folder"
fi


# Dynamically set the filename based on the dataset
output_file="results/${folder}/${datasets[0]}.csv"

# Loop through all combinations and run the script
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
	for round in "${rounds[@]}"; do
	    for init in "${cls_inits[@]}"; do
	        for seed in "${seeds[@]}"; do
	            echo "Running: $dataset $method $init $round $seed"
	            # Run the script and capture the output
	            output=$(python train_lp_ft_ct.py --dataset "$dataset" --method "$method" --cls_init "$init" --round "$round" --seed "$seed" --epochs "$epochs" --bsz "$batch_size" --ALMETHOD "$ALMETHOD" --log_mode "$log_mode" --model_cfg "$model_cfg" --folder "$output_folder")
	            # Print the output to the console
                    echo "$output"
                    # Append the results to the CSV file
                    echo "$output" >> "$output_file"
                done
            done
        done
    done
done
