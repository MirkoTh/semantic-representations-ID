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
    "device": "cpu"  # "cuda:0"
}

# Define the variables and their possible values
# lmbda_list = [0.0005, 0.001]
# embed_dim_list = [10]
# agreement_list = ["most", "few"]
# sparsity_list = ["ID", "both"]

lmbda_list = [0.01, 0.001, 0.0001]
# embed_dim_list = [5]
agreement_list = ["few", "most"]
# sparsity_list = ["ID"]
learning_rate_list = [0.001, 0.01]

# Generate all combinations
combinations = list(itertools.product(
    # , sparsity_list)), embed_dim_list
    lmbda_list, agreement_list, learning_rate_list))

# Create the list of dictionaries
arg_combinations = []
# , sparsity, embed_dim in combinations:
for lmbda, agreement, learning_rate in combinations:
    temp_dict = base_dict.copy()
    temp_dict.update({
        'lmbda': lmbda,
        # 'embed_dim': embed_dim,
        'agreement': agreement,
        # 'sparsity': sparsity,
        'learning_rate': learning_rate
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
        --agreement {args['agreement']} \
        --steps {args['steps']} \
        --device {args['device']}"
    )
    subprocess.run(command, shell=True)
    #
    # --sparsity {args['sparsity']} \
    # --embed_dim {args['embed_dim']} "


# for args in arg_combinations:
#     run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(run_command, arg_combinations)
