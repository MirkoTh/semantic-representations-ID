import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

import os

os.environ["USE_OPENMP"] = "1"
os.environ["MKL_THREADING_LAYER"] = "TBB"

# Define the fixed parts of the dictionary
base_dict = {
    "triplets_dir": "./data",
    "task": "odd_one_out",
    "epochs": 100,
    "steps": 25,
    "device": "cpu",  # "cuda:0" #
}

# combined model always uses
# random by-participant decision weights
# lmbda = 0.0005
# lmbda_hierarchical = 0.01
# fix coded in run-embedding-decision-combined-data.py

l_rnd_seed = [1]  # 1, 2, 3, 4, 5, , , 11, 12, 13, 14, 15
embed_dim_list = [200]  # 15,2, 3, 4, 5, 6 , 10, 15, 50, 85, 25, 50
learning_rate_list = [0.0005]  # 0.0005
l_lmbda = [0.0000001, 0.008, 0.01, 0.0105, 0.011, 0.012, 0.0125, 0.015, 0.025, 0.05, 0.1, 0.5, 0.9] #[0.0000001]  # 0.0000001, , 0.008, 0.05, 0.1, 0.5, 0.9, 0.0005
# , "random_weights_random_scaling"
modeltype_list = ["free_weights_no_scaling", "free_weights_free_scaling"]#
l_python_file = ["run-embedding-decision-combined-data.py"]

# dataset explanations
# "testcase": small dataset for testing the code
# "full": full dataset, including data from new batch (2025-08)
# "first_half": first half of the full dataset, including data from new batch (2025-08)
# "second_half": second half of the full dataset, including data from new batch (2025-08)
# "full_evaluate_actual": Hebart et al. (2023), but selected subset with correct subject ID, 90/10 train-test split
# "full_evaluate_shuffled": Hebart et al. (2023), but selected subset with shuffled subject ID, 90/10 train-test split
l_data_subset = ["full_evaluate_actual", "full_evaluate_shuffled"]  # ,"testcase", , "full", "first_half", "second_half", "first_half_v2", "second_half_v2"
l_individual_slopes_type = ["separate"]  # ,  ,"shared",, "shared_and_separate"

# Generate all combinations
combinations = list(
    itertools.product(
        l_rnd_seed, l_lmbda, learning_rate_list, embed_dim_list, modeltype_list, 
        l_python_file, l_data_subset, l_individual_slopes_type
    )
)

# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for (
    rnd_seed,
    lmbda,
    learning_rate,
    embed_dim,
    modeltype,
    python_file,
    data_subset,
    individual_slopes_type,
) in combinations:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update(
        {
            "rnd_seed": rnd_seed,
            "lmbda": lmbda,
            "learning_rate": learning_rate,
            "embed_dim": embed_dim,
            "modeltype": modeltype,
            "python_file": python_file,
            "data_subset": data_subset,
            "individual_slopes_type": individual_slopes_type,
        }
    )
    arg_combinations.append(temp_dict)


# Function to run the command


def run_command(args):
    command = f" python {args['python_file']} --rnd_seed {args['rnd_seed']} \
        --lmbda {args['lmbda']} \
        --triplets_dir {args['triplets_dir']} \
        --task {args['task']} \
        --learning_rate {args['learning_rate']} \
        --modeltype {args['modeltype']} \
        --epochs {args['epochs']} \
        --embed_dim {args['embed_dim']} \
        --data_subset {args['data_subset']} \
        --individual_slopes_type {args['individual_slopes_type']} \
        --steps {args['steps']} \
        --device {args['device']}"
    subprocess.run(command, shell=True)


for args in arg_combinations:
    run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
# with ThreadPoolExecutor(max_workers=18) as executor:
#     executor.map(run_command, arg_combinations)
