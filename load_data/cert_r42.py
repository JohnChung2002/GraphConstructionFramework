import pyarrow.feather as feather
import pandas as pd
import pickle
import glob
from tqdm import tqdm
import csv

"""
paths = {
    'name1': { 'path': '/path1', 'type': 'feather' },
}

type can be (csv, pickle, feather)
"""

def load_data(data_info: dict):
    data = {}
    for key, value in data_info.items():
        if key == 'answers':
            continue
        if value['type'] == 'feather':
            with open(value['path'], 'rb') as f:
                data[key] = feather.read_feather(f)
        if value['type'] == 'csv':
            data[key] = pd.read_csv(value['path'])
        if value['type'] == 'pickle':
            with open(value['path'], 'rb') as f:
                data[key] = pickle.load(f)
    data["answers"] = load_answers(data_info['answers'])
    return data

def process_answers_row(row):
    data = {}
    data['log_type'] = row[0]
    data['id'] = row[1]
    data['date'] = row[2]
    data['user_id'] = row[3]
    data['pc'] = row[4]
    if row[0] == 'logon' or row[0] == 'device':
        data['activity'] = row[5]
    elif row[0] == 'http':
        data['url'] = row[5]
        data['content'] = row[6]
    elif row[0] == 'file':
        data['filename'] = row[5]
        data['content'] = row[6]
    elif row[0] == 'email':
        data['to'] = row[5]
        data['cc'] = row[6]
        data['bcc'] = row[7]
        data['from'] = row[8]
        data['size'] = row[9]
        data['attachments'] = row[10]
    return data

def load_answers(paths: list[str]): 
    answers = pd.DataFrame(columns=['log_type', 'id', 'date', 'user_id', 'pc', 'activity', 'url', 'filename', 'content', 'to', 'cc', 'bcc', 'from', 'size', 'attachments'])
    temp_paths = []
    for path in paths:
        temp_paths.append(glob.glob(f'{path}/*.csv'))
    scenario_file_paths = [item for sublist in temp_paths for item in sublist]
    for file in tqdm(scenario_file_paths, desc="Processing Answers"):
        with open(file, 'r') as f:
            read = csv.reader(f)
            next(read)
            for i in read:
                data = process_answers_row(i)
                answers = answers._append(data, ignore_index=True)

    answers['date'] = pd.to_datetime(answers['date'], format="%m/%d/%Y %H:%M:%S")
    answers.sort_values(by=['date'], inplace=True)
    answers.reset_index(drop=True, inplace=True)
    return answers