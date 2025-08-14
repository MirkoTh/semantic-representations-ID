import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

import os

os.environ["USE_OPENMP"] = "1"
os.environ["MKL_THREADING_LAYER"] = "TBB"

# Define the fixed parts of the dictionary
base_dict = {
    "rnd_seed": 7640,
    "triplets_dir": "./data/study1-2025-08",
    "task": "odd_one_out",
    "epochs": 10,
    "steps": 5,
    "device": "cpu",  # "cuda:0" #
}

# combined model always uses
# random by-participant decision weights
# lmbda = 0.0005
# lmbda_hierarchical = 0.01
# fix coded in run-embedding-decision-combined-data.py

embed_dim_list = [7]  # 15
learning_rate_list = [0.0005]  # 0.0005
modeltype_list = ["random_weights_random_scaling"]
python_file = ["run-embedding-decision-combined-data.py"]

# Generate all combinations
combinations = list(
    itertools.product(
        learning_rate_list,
        embed_dim_list,
        modeltype_list,
        python_file,
    )
)

# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for (
    learning_rate,
    embed_dim,
    modeltype,
    python_file,
) in combinations:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update(
        {
            "learning_rate": learning_rate,
            "embed_dim": embed_dim,
            "modeltype": modeltype,
            "python_file": python_file,
        }
    )
    arg_combinations.append(temp_dict)


# Function to run the command


def run_command(args):
    command = f" python {args['python_file']} --rnd_seed {args['rnd_seed']} \
        --triplets_dir {args['triplets_dir']} \
        --task {args['task']} \
        --learning_rate {args['learning_rate']} \
        --modeltype {args['modeltype']} \
        --epochs {args['epochs']} \
        --embed_dim {args['embed_dim']} \
        --steps {args['steps']} \
        --device {args['device']}"
    subprocess.run(command, shell=True)


# for args in arg_combinations:
#     run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_command, arg_combinations)
