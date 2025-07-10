import json
import pandas as pd

questionnaire_data = pd.read_csv('experiments\questionnaire-setup\questionnaires_info.csv', sep = ";")

data = {}

for i in questionnaire_data.Measure.unique():
    data[i] = dict(questions=[], preamble=questionnaire_data.Preamble[questionnaire_data.Measure == i].iloc[0],
                   name=i)
    for j in questionnaire_data[questionnaire_data.Measure == i].iterrows():
        print([i for i in j[1].Options.split(',') if len(i)])
        data[i]['questions'].append(dict(prompt=j[1].Question,
                                            labels=[i for i in j[1].Options.split(',') if len(i)]))

json_data = json.dumps(data)

with open('experiments\questionnaire-setup\questionnaires.json', 'w') as outfile:
    json.dump(data, outfile)

