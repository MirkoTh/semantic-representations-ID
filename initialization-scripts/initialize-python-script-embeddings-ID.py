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



lmbda_list = [0, 0.000025]
# embed_dim_list = [15, 50]
# agreement_list = ["few", "most"]
# sparsity_list = ["ID"]
learning_rate_list = [0.1, 0.05]

# Generate all combinations
combinations = list(itertools.product(
    # , sparsity_list)), agreement_list
    lmbda_list, learning_rate_list))

# Create the list of dictionaries
arg_combinations = []
# , sparsity, embed_dim in combinations:
for lmbda, learning_rate in combinations: #, agreement
    temp_dict = base_dict.copy()
    temp_dict.update({
        'lmbda': lmbda,
        'learning_rate': learning_rate,
        #'embed_dim': embed_dim,
        #'agreement': agreement,
        # 'sparsity': sparsity,
    })
    arg_combinations.append(temp_dict)

# Path to the Python file you want to run
python_file = 'python/run-ID-on-embeddings.py'

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
        --device {args['device']}"
    )
    subprocess.run(command, shell=True)
    #
    # --sparsity {args['sparsity']} \
    # --agreement {args['agreement']} \
    # --embed_dim {args['embed_dim']} \



for args in arg_combinations:
    run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
# with ThreadPoolExecutor(max_workers=2) as executor:
#     executor.map(run_command, arg_combinations)
