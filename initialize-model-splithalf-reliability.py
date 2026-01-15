import subprocess
from concurrent.futures import ThreadPoolExecutor

import itertools

import os

os.environ["USE_OPENMP"] = "1"
os.environ["MKL_THREADING_LAYER"] = "TBB"

# Define the fixed parts of the dictionary
base_dict = {
    "triplets_dir": "./data",
    "task": "odd_one_out",
    "epochs": 100,
    "steps": 25,
    "device": "cpu",  # "cuda:0" #
}

learning_rate_list = [0.0005]  # 0.0005
l_lmbda = [0.0000001] #[0.0000001]  # 0.0000001, , 0.008, 0.05, 0.1, 0.5, 0.9, 0.0005
modeltype_list = ["free_weights_no_scaling"]#, "free_weights_free_scaling"
l_python_file = ["run-embedding-decision-combined-data.py"]
l_data_subset = ["first_half", "second_half", "first_half_v2", "second_half_v2"]  # ,"full" , "full_evaluate_actual", "full_evaluate_shuffled","testcase", , "full", "first_half", "second_half", "first_half_v2", "second_half_v2"
l_individual_slopes_type = ["separate"]  # ,  ,"shared",, "shared_and_separate"

l_rnd_seed = [1, 2, 3]  # 1, 2, 3, 4, 5, 11, 12, 13, 14, 15
embed_dim_list = [2, 3, 4, 5, 6] # 15, 2, 3, 4, 5, 6, 10, 15, 50, 85, 25, 50

# Generate all combinations
combinations = list(
    itertools.product(
        l_rnd_seed, l_lmbda, learning_rate_list, embed_dim_list, modeltype_list, 
        l_python_file, l_data_subset, l_individual_slopes_type
    )
)

l_rnd_seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
embed_dim_list = [7, 8, 9]
# Generate all combinations
combinations2 = list(
    itertools.product(
        l_rnd_seed, l_lmbda, learning_rate_list, embed_dim_list, modeltype_list, 
        l_python_file, l_data_subset, l_individual_slopes_type
    )
)

l_rnd_seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
embed_dim_list = [10, 15]

combinations3 = list(
    itertools.product(
        l_rnd_seed, l_lmbda, learning_rate_list, embed_dim_list, modeltype_list, 
        l_python_file, l_data_subset, l_individual_slopes_type
    )
)

l_rnd_seed = [1]
l_embed_dim = [25, 35]

combinations4 = list(
    itertools.product(
        l_rnd_seed, l_lmbda, learning_rate_list, embed_dim_list, modeltype_list, 
        l_python_file, l_data_subset, l_individual_slopes_type
    )
)

combinations_all = combinations + combinations2 + combinations3 + combinations4
# Create the list of dictionaries
arg_combinations = []
#  in combinations:
for (
    rnd_seed,
    lmbda,
    learning_rate,
    embed_dim,
    modeltype,
    python_file,
    data_subset,
    individual_slopes_type,
) in combinations_all:  # , agreement
    temp_dict = base_dict.copy()
    temp_dict.update(
        {
            "rnd_seed": rnd_seed,
            "lmbda": lmbda,
            "learning_rate": learning_rate,
            "embed_dim": embed_dim,
            "modeltype": modeltype,
            "python_file": python_file,
            "data_subset": data_subset,
            "individual_slopes_type": individual_slopes_type,
        }
    )
    arg_combinations.append(temp_dict)


# Function to run the command


def run_command(args):
    command = f" python {args['python_file']} --rnd_seed {args['rnd_seed']} \
        --lmbda {args['lmbda']} \
        --triplets_dir {args['triplets_dir']} \
        --task {args['task']} \
        --learning_rate {args['learning_rate']} \
        --modeltype {args['modeltype']} \
        --epochs {args['epochs']} \
        --embed_dim {args['embed_dim']} \
        --data_subset {args['data_subset']} \
        --individual_slopes_type {args['individual_slopes_type']} \
        --steps {args['steps']} \
        --device {args['device']}"
    subprocess.run(command, shell=True)


for args in arg_combinations:
    run_command(args)
# Use ThreadPoolExecutor to run the commands in parallel
# with ThreadPoolExecutor(max_workers=18) as executor:
#     executor.map(run_command, arg_combinations)
