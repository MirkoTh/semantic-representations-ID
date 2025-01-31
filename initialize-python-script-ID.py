import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

# Define the fixed parts of the dictionary
base_dict = {
    'rnd_seed': 852,
    'triplets_dir': './data/',
    "task": "odd_one_out",
    "epochs": 250,
    "steps": 125,
    "device": "cuda:0"  # "cuda:0"
}

# Define the variables and their possible values
# lmbda_list = [0.0005, 0.001]
# embed_dim_list = [10]
# agreement_list = ["most", "few"]
# sparsity_list = ["ID", "both"]

lmbda_list = [0.0005]
embed_dim_list = [15]
agreement_list = ["few"]
sparsity_list = ["ID", "both"]
learning_rate_list = [0.0005]
id_weights_only_list = [False]

# Generate all combinations
combinations = list(itertools.product(
    # ))
    lmbda_list, learning_rate_list, embed_dim_list, agreement_list, sparsity_list, id_weights_only_list))

# Create the list of dictionaries
# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for lmbda, learning_rate, embed_dim, agreement, sparsity, id_weights_only in combinations: #, agreement
    temp_dict = base_dict.copy()
    temp_dict.update({
        'lmbda': lmbda,
        'learning_rate': learning_rate,
        'embed_dim': embed_dim,
        'agreement': agreement,
        'sparsity': sparsity,
        'id_weights_only': id_weights_only
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
        --lmbda {args['lmbda']} \
        --sparsity {args['sparsity']} \
        --agreement {args['agreement']} \
        --id_weights_only {args['id_weights_only']} \
        --epochs {args['epochs']} \
        --embed_dim {args['embed_dim']} \
        --steps {args['steps']} \
        --device {args['device']}"
    )
    subprocess.run(command, shell=True)
    #

for args in arg_combinations:
    print("id_weights_only = " + args['id_weights_only'])

# for args in arg_combinations:
#     run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
# with ThreadPoolExecutor(max_workers=2) as executor:
#     executor.map(run_command, arg_combinations)
