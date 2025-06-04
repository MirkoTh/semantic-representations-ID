import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

# Define the fixed parts of the dictionary
base_dict = {
    'rnd_seed': 852,
    'triplets_dir': './data/',
    "task": "odd_one_out",
    "epochs": 10,
    "steps": 5,
    "device": "cpu"  # "cuda:0"
}

lmbda_list = [0.0005]
lmbda_hierarchical_list = [.01, .1, 1]
embed_dim_list = [15] #15
sparsity_list = ["items_and_random_ids"] #, "both"
learning_rate_list = [0.0005]
modeltype_list = ["random_weights_free_scaling"]
splithalf_list = ["1", "2", "no"]

# Generate all combinations
combinations = list(itertools.product(
    lmbda_list, lmbda_hierarchical_list, learning_rate_list, 
    embed_dim_list, sparsity_list, modeltype_list,
    splithalf_list
    ))

# Create the list of dictionaries
# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for lmbda, lmbda_hierarchical, learning_rate, embed_dim, sparsity, modeltype, splithalf in combinations:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update({
        'lmbda': lmbda,
        'lmbda_hierarchical': lmbda_hierarchical,
        'learning_rate': learning_rate,
        'embed_dim': embed_dim,
        'sparsity': sparsity,
        'modeltype': modeltype,
        'splithalf': splithalf
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
        --lmbda_hierarchical {args['lmbda_hierarchical']} \
        --sparsity {args['sparsity']} \
        --modeltype {args['modeltype']} \
        --splithalf {args['splithalf']} \
        --epochs {args['epochs']} \
        --embed_dim {args['embed_dim']} \
        --steps {args['steps']} \
        --device {args['device']}"
    )
    subprocess.run(command, shell=True)
    #


for args in arg_combinations:
    run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
#with ThreadPoolExecutor(max_workers=3) as executor:
#    executor.map(run_command, arg_combinations)
