"""
Recommended to load data and answers into dataframes
load_data should return a dictionary of dataframes (data and answers) and additional data required in subsquent steps

Data returned from load_data would be passed into preprocess_data
"""

def load_answers(answer_paths: dict):
    answers = {}
    for key, value in answer_paths.items():
        # load answers
        pass
    return answers

def load_data(data_paths: dict):
    data = {}
    for key, value in data_paths.items():
        # load data
        pass
    answers = load_answers(data_paths["answers"])
    return { "data": data, "answers": answers }