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
    "use_shuffled_subjects": "shuffled",
    "early_stopping": "Yes"
}

# Define the variables and their possible values
# lmbda_list = [0.0005, 0.001]
# embed_dim_list = [10]

lmbda_list = [0.01, 0.0005]
embed_dim_list = [15]
learning_rate_list = [0.0005]

# Generate all combinations
combinations = list(itertools.product(
    lmbda_list, embed_dim_list, learning_rate_list))

# Create the list of dictionaries
arg_combinations = []
# , sparsity in combinations:
for lmbda, embed_dim, learning_rate in combinations:
    temp_dict = base_dict.copy()
    temp_dict.update({
        'lmbda': lmbda,
        'embed_dim': embed_dim,
        'learning_rate': learning_rate
    })
    arg_combinations.append(temp_dict)

# Path to the Python file you want to run
python_file = 'run-avg-ID-jointly.py'

# Function to run the command


def run_command(args):
    command = (
        f" python {python_file} --rnd_seed {args['rnd_seed']} \
        --triplets_dir {args['triplets_dir']} \
        --task {args['task']} \
        --learning_rate {args['learning_rate']} \
        --embed_dim {args['embed_dim']} \
        --lmbda {args['lmbda']} \
        --epochs {args['epochs']} \
        --steps {args['steps']} \
        --device {args['device']} \
        --early_stopping {args['early_stopping']} \
        --use_shuffled_subjects {args['use_shuffled_subjects']} "
    )
    subprocess.run(command, shell=True)
    # --embed_dim {args['embed_dim']} "


# for args in arg_combinations:
#     run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(run_command, arg_combinations)
