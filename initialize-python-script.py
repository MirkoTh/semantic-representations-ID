import subprocess

# Define the different combinations of named arguments
arg_combinations = [
    {'rnd_seed': 4312, 'triplets_dir': './data/', "task": "odd_one_out", "learning_rate": .001, "lmbda": 0.01, "epochs": 100, "embed_dim": 5},
    {'rnd_seed': 4312, 'triplets_dir': './data/', "task": "odd_one_out", "learning_rate": .001, "lmbda": 0.01, "epochs": 100, "embed_dim": 10},
    {'rnd_seed': 4312, 'triplets_dir': './data/', "task": "odd_one_out", "learning_rate": .001, "lmbda": 0.01, "epochs": 100, "embed_dim": 20},
    {'rnd_seed': 4312, 'triplets_dir': './data/', "task": "odd_one_out", "learning_rate": .001, "lmbda": 0.01, "epochs": 100, "embed_dim": 30},
    {'rnd_seed': 4312, 'triplets_dir': './data/', "task": "odd_one_out", "learning_rate": .001, "lmbda": 0.01, "epochs": 100, "embed_dim": 40},
    {'rnd_seed': 4312, 'triplets_dir': './data/', "task": "odd_one_out", "learning_rate": .001, "lmbda": 0.01, "epochs": 100, "embed_dim": 50},
]

# Path to the Python file you want to run
python_file = 'run-avg-ID.py'

# Iterate over the combinations and run the commands
for args in arg_combinations:
    command = (
        f" python {python_file} --rnd_seed {args['rnd_seed']} \
        --triplets_dir {args['triplets_dir']} \
        --task {args['task']} \
        --learning_rate {args['learning_rate']} \
        --lmbda {args['lmbda']} \
        --epochs {args['epochs']} \
        --embed_dim {args['embed_dim']} "\
    )
    subprocess.run(command, shell=True)
