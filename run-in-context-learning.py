import torch
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unsloth import FastLanguageModel

from os.path import join as pjoin
import os
from functools import partial
import re
from tqdm.notebook import tqdm
import pickle

nbilly = 8
engine = f"""meta-llama/Llama-3.1-{nbilly}B-Instruct"""
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=engine,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto"
)
tokenizer.pad_token = tokenizer.bos_token
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

FastLanguageModel.for_inference(model)


device = "cuda:0"
model = model.to(device)
model.parallelize()


results_dir = f"""semantic-representations-ID/results/in-context-learning/lama-3.1-{nbilly}B"""
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


train_triplets = torch.from_numpy(np.loadtxt(
    pjoin("./semantic-representations-ID/data/", 'train_90_ID.txt'))).to(device).type(torch.LongTensor)
triplet_ids = train_triplets.detach().numpy()
tbl_labels = pd.read_csv(
    "./semantic-representations-ID/data/unique_id.txt", delimiter="\\", header=None)
tbl_labels[0] = tbl_labels[0].str.replace(r'\d+', '', regex=True)


def randomize_and_extract_trial(tbl_labels, l_ids):
    tbl_select = tbl_labels.iloc[l_ids[0:3]]
    tbl_select.reset_index(drop=True, inplace=True)
    rand_order = np.random.choice([0, 1, 2], 3, replace=False)
    tbl_reordered = tbl_select.iloc[rand_order].reset_index(drop=True)
    tbl_reordered["rnd"] = rand_order
    tbl_return = pd.DataFrame({
        "1": [tbl_reordered.iloc[0, 0]],
        "2": [tbl_reordered.iloc[1, 0]],
        "3": [tbl_reordered.iloc[2, 0]],
        "selected": np.where(tbl_reordered["rnd"] == 2)[0] + 1
    })
    return tbl_return


randomize_and_extract_partial = partial(
    randomize_and_extract_trial, tbl_labels)


# first start with an equal number of trials for each participant
# this can be changed later to the actual number of trials for each participant
n_trials = 100

# here one participant is selected
# this can be put into a function be called for each participant

l_dfs_results = []

for participant_id in range(0, 3):

    l_trials = list(map(randomize_and_extract_partial,
                    triplet_ids[triplet_ids[:, 3] == participant_id]))

    df_trials = pd.concat(l_trials)

    l_all_current_tasks = []
    l_all_previous_trials = []
    for current_trial_id in range(0, n_trials):
        l_previous_trials = []
        current_trial = df_trials.iloc[current_trial_id, :]
        current_task = ", ".join(
            [f"{index+1}. {item}" for index, item in enumerate(current_trial[0:3])])
        if current_trial_id > 0:
            for previous_trial_id in range(0, (current_trial_id)):
                previous_trial = df_trials.iloc[previous_trial_id, :]
                response_prev = f"""{previous_trial["selected"]}. {previous_trial.iloc[previous_trial["selected"]-1]} in: """ + ", ".join(
                    [f"{index+1}. {item}" for index, item in enumerate(previous_trial[0:3])])
                l_previous_trials.append(response_prev)
        l_all_current_tasks.append(current_task)
        l_all_previous_trials.append(l_previous_trials)

    system_input = (
        "You are a cognitive scientist and you are given a by-trial data set from a participant. " +
        "in the odd-one-out task. For each trial, you are presented with the three words and " +
        "the participant's response, i.e., the word the participant thinks is the odd one out of the three. " +

        "\nYour goal is to predict, for a new triplet of words, " +
        "which word the person is most likely to identify as the odd one out." +
        "Note that in the very first trial, you are only presented with the three words, but no previous responses."

    )
    input_text = (
        "In an odd-one-out task, a participant is presented with three objects in each trial.\n" +
        "The participant is instructed to select the object they think is most dissimilar from the other presented objects.\n" +
        "A participant's response is neither correct nor incorrect. The response should simply reflect the participant's " +
        "perceived semantic similarity between the three objects."
    )

    l_responses_int = []

    for trial_id in tqdm(range(0, n_trials)):
        l_prev_concat = l_all_previous_trials[trial_id]
        current = l_all_current_tasks[trial_id]

        if trial_id == 0:
            question_text = (
                "\nWhen reasoning what logic the person used, think in steps - for example:\n"
                "is a participant sensitive to a set of dimensions in semantic space (e.g., technology, food, animals, ...)?"
                "how would this person respond given three new objects?\n"
                f"""\nThe current trial is trial nr. {trial_id + 1}"""
                "\n\nPlease first respond with the respective number (i.e., 1., 2., or 3.), and only then explain your reasoning.\n"
                "Make sure that the number corresponds with your reasoning. For example, if you respond with 2. in the triplet\n"
                "1. house, 2. zebra, 3. garden, then the number 2. should only refer to 2. zebra\n"
                "\nWhich of the following three words is the person to denote as the odd-one-out?\n"
                f"""{current}?"""
            )
        elif trial_id > 0:
            question_text = (
                f"""\nThe current trial is trial nr. {trial_id + 1}"""
                "\nWhen reasoning what logic the person used, think in steps - for example:\n"
                "is a participant sensitive to a set of dimensions in semantic space (e.g., technology, food, animals, ...)?"
                "how would this person respond given three new objects?\n"
                "\nHere are the previous responses from the participant:\n" +
                "\n".join(l_prev_concat) +
                "\n\nPlease first respond with the respective number (i.e., 1., 2., or 3.), and only then explain your reasoning.\n"
                "Make sure that the number corresponds with your reasoning. For example, if you respond with 2. in the triplet\n"
                "1. house, 2. zebra, 3. garden, then the number 2. should only refer to 2. zebra\n"
                "\nWhich of the following three words is the person to denote as the odd-one-out?\n"
                f"""{current}?"""
            )

        user_input = input_text + question_text

        messages = [
            {"role": "system", "content": system_input},
            {"role": "user", "content": user_input},
            {"role": "system", "content": "For the newest triplet, the participant responds with Nr. "}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            # add_generation_prompt=True,
            return_tensors="pt",
            continue_final_message=True
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=2,
            eos_token_id=terminators,
            do_sample=False,
            # temperature=0.000001
        )

        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)
        resp_extract = re.search("[1-3]", decoded_response)

        resp_try = resp_extract.group()

        try:
            resp_int = int(resp_try)
        except:
            resp_int = np.nan

        l_responses_int.append(resp_int)

    # add the lama responses to the dataframe
    df_trials_pred = df_trials.head(n_trials).copy()
    df_trials_pred["predicted"] = l_responses_int
    df_trials_pred.query(
        "selected == predicted").shape[0] / df_trials_pred.shape[0]
    df_trials_pred["accuracy"] = df_trials_pred["selected"] == df_trials_pred["predicted"]
    df_trials_pred["accuracy"] = df_trials_pred["accuracy"].astype(int)
    l_dfs_results.append(df_trials_pred)

    pth_file = os.path.join(results_dir, "results.pkl")
    with open(pth_file, "wb") as f:
        pickle.dump(l_dfs_results, f)
