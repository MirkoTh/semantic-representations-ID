import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

# Define the fixed parts of the dictionary
base_dict = {
    'rnd_seed': 852,
    'triplets_dir': './data/',
    "task": "odd_one_out",
    "epochs": 1500,
    "steps": 250,
    "device": "cpu"  # "cuda:0"
}

# Define the variables and their possible values
# lmbda_list = [0.0005, 0.001]
# embed_dim_list = [10]
# agreement_list = ["most", "few"]
# sparsity_list = ["ID", "both"]

lmbda_list = [0.01]  # [0.008, 0.0005]
embed_dim_list = [100]  # [15, 50, 100]
# agreement_list = ["few", "most"]
# sparsity_list = ["ID"]
learning_rate_list = [0.0005]

# Generate all combinations
combinations = list(itertools.product(
    # , sparsity_list)), agreement_list
    lmbda_list, learning_rate_list, embed_dim_list))

# Create the list of dictionaries
# Create the list of dictionaries
arg_combinations = []
# , sparsity, embed_dim in combinations:
for lmbda, learning_rate, embed_dim in combinations:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update({
        'learning_rate': learning_rate,
        'lmbda': lmbda,
        'embed_dim': embed_dim,
        # 'agreement': agreement,
        # 'sparsity': sparsity,
    })
    arg_combinations.append(temp_dict)

# Path to the Python file you want to run
python_file = 'run-avg-only.py'

# Function to run the command


def run_command(args):
    command = (
        f" python {python_file} --task {args['task']} \
        --triplets_dir {args['triplets_dir']} \
        --learning_rate {args['learning_rate']} \
        --lmbda {args['lmbda']} \
        --embed_dim {args['embed_dim']} \
        --epochs {args['epochs']} \
        --steps {args['steps']} \
        --device {args['device']} \
        --rnd_seed {args['rnd_seed']}"

    )

    subprocess.run(command, shell=True)
    #
    # --sparsity {args['sparsity']} \
    # --agreement {args['agreement']} \


# for args in arg_combinations:
#     run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
with ThreadPoolExecutor(max_workers=1) as executor:
    executor.map(run_command, arg_combinations)
