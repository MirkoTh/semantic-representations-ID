import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

# Define the fixed parts of the dictionary
base_dict = {
    'rnd_seed': 852,
    'triplets_dir': './data/',
    "task": "odd_one_out",
    "epochs": 10,
    "steps": 10,
    "device": "cpu",  # "cuda:0"
    # "use_shuffled_subjects": "shuffled",
    "early_stopping": "Yes"
}

# Define the variables and their possible values
# lmbda_list = [0.0005, 0.001]

lmbda_list = [0.001, 0.0005]
learning_rate_list = [0.0005]
use_shuffled_subjects_list = ["shuffled", "actual"]

# Generate all combinations
combinations = list(itertools.product(
    lmbda_list, learning_rate_list, use_shuffled_subjects_list))

# Create the list of dictionaries
arg_combinations = []
# , sparsity in combinations:
for lmbda, learning_rate, use_shuffled_subjects in combinations:
    temp_dict = base_dict.copy()
    temp_dict.update({
        'lmbda': lmbda,
        'learning_rate': learning_rate,
        'use_shuffled_subjects': use_shuffled_subjects
    })
    arg_combinations.append(temp_dict)

# Path to the Python file you want to run
python_file = 'run-ID-on-embeddings.py'

# Function to run the command


def run_command(args):
    command = (
        f" python {python_file} --rnd_seed {args['rnd_seed']} \
        --triplets_dir {args['triplets_dir']} \
        --task {args['task']} \
        --learning_rate {args['learning_rate']} \
        --lmbda {args['lmbda']} \
        --epochs {args['epochs']} \
        --steps {args['steps']} \
        --device {args['device']} \
        --early_stopping {args['early_stopping']} \
        --use_shuffled_subjects {args['use_shuffled_subjects']} "
    )
    subprocess.run(command, shell=True)


# for args in arg_combinations:
#     run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_command, arg_combinations)
