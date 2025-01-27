import subprocess
# from concurrent.futures import ThreadPoolExecutor

import itertools

# Define the fixed parts of the dictionary
base_dict = {
    'rnd_seed': 549,
    'triplets_dir': './data/',
    "task": "odd_one_out",
    "learning_rate": .0005,
    "epochs": 2,
    "steps": 10,
    "device": "cpu"  # "cuda:0"
}

# Define the variables and their possible values
# lmbda_list = [0.0005, 0.001]
# embed_dim_list = [10]
# agreement_list = ["many", "few"]
# sparsity_list = ["ID", "both"]

lmbda_list = [0.001]
embed_dim_list = [5]
# agreement_list = ["few"]
# sparsity_list = ["ID"]

# Generate all combinations
combinations = list(itertools.product(
    lmbda_list, embed_dim_list))  # agreement_list, sparsity_list))

# Create the list of dictionaries
arg_combinations = []
for lmbda, embed_dim in combinations:  # agreement, sparsity in combinations:
    temp_dict = base_dict.copy()
    temp_dict.update({
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
        f" python {python_file} --rnd_seed {args['rnd_seed']} \
        --triplets_dir {args['triplets_dir']} \
        --task {args['task']} \
        --learning_rate {args['learning_rate']} \
        --lmbda {args['lmbda']} \
        --epochs {args['epochs']} \
        --steps {args['steps']} \
        --device {args['device']} \
        --embed_dim {args['embed_dim']} "
    )
    subprocess.run(command, shell=True)
    # --agreement {args['agreement']} \
    # --sparsity {args['sparsity']} \


for args in arg_combinations:
    run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
# with ThreadPoolExecutor(max_workers=24) as executor:
#    executor.map(run_command, arg_combinations)
