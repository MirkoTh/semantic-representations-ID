import subprocess
from concurrent.futures import ThreadPoolExecutor

# Define the different combinations of named arguments
arg_combinations = [
    {'rnd_seed': 1234, 'triplets_dir': './data/', "task": "odd_one_out",
        "learning_rate": .001, "lmbda": 0.01, "epochs": 5, "embed_dim": 200, "loggername":"lmbd=.01;ndim=200"},
    {'rnd_seed': 4312, 'triplets_dir': './data/', "task": "odd_one_out",
        "learning_rate": .001, "lmbda": 0.001, "epochs": 5, "embed_dim": 200, "loggername":"lmbd=.01;ndim=200"},
    {'rnd_seed': 4356, 'triplets_dir': './data/', "task": "odd_one_out",
        "learning_rate": .001, "lmbda": 0.1, "epochs": 5, "embed_dim": 200, "loggername":"lmbd=.01;ndim=200"},
    {'rnd_seed': 87456, 'triplets_dir': './data/', "task": "odd_one_out",
        "learning_rate": .001, "lmbda": 0.01, "epochs": 5, "embed_dim": 400, "loggername":"lmbd=.01;ndim=200"},
    {'rnd_seed': 2345, 'triplets_dir': './data/', "task": "odd_one_out",
        "learning_rate": .001, "lmbda": 0.001, "epochs": 5, "embed_dim": 400, "loggername":"lmbd=.01;ndim=200"},
    {'rnd_seed': 45, 'triplets_dir': './data/', "task": "odd_one_out",
        "learning_rate": .001, "lmbda": 0.1, "epochs": 5, "embed_dim": 400, "loggername":"lmbd=.01;ndim=200"},
]

# Path to the Python file you want to run
python_file = 'run-avg-ID-jointly.py'

# Function to run the command


def run_command(args):
    command = (
        f" python {python_file} --rnd_seed {args['rnd_seed']} \
        --loggername {args['loggername']} \
        --triplets_dir {args['triplets_dir']} \
        --task {args['task']} \
        --learning_rate {args['learning_rate']} \
        --lmbda {args['lmbda']} \
        --epochs {args['epochs']} \
        --embed_dim {args['embed_dim']} "
    )
    subprocess.run(command, shell=True)


# Use ThreadPoolExecutor to run the commands in parallel
with ThreadPoolExecutor(max_workers=6) as executor:
    executor.map(run_command, arg_combinations)
