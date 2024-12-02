import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

if "/fred/oz382/nltk" not in nltk.data.path:
    nltk.data.path.append("/fred/oz382/nltk")

def process_log_type(log_type, data, x_data, y_data, encodings):
    """
    General function to process log data for different types.
    Args:
    - log_type (str): Type of log ('logon', 'device', 'file', 'http')
    - data (DataFrame): Data for the specific log type
    - x_data (dict): Dictionary to store input features
    - y_data (dict): Dictionary to store labels
    - encodings (dict): Dictionary of necessary encodings (e.g., user_encodings, pc_encodings, etc.)
    """
    for index, row in data.iterrows():
        # Prepare the feature vector
        feature_vector = [
            row['date'].timestamp(),  # Timestamp
            encodings["user"][row['user']],  # User encoding
            encodings["pc"][row['pc']],  # PC encoding
            row['off_work'],  # Off work status
        ]
        
        # Add log type specific features
        if log_type == "logon":
            feature_vector.append(encodings["logon_activity"][row['activity']])  # Logon activity encoding
        elif log_type == "device":
            feature_vector.append(encodings["device_activity"][row['activity']])  # Device activity encoding
        elif log_type == "file":
            feature_vector.extend([
                encodings["filename"][row['filename']],  # Filename encoding
                encodings["filetype"][row['filetype']],  # File type encoding
                row['sentiment']  # Sentiment
            ])
        elif log_type == "http":
            feature_vector.extend([
                encodings["domain"][row['domain']],  # Domain encoding
                row['keylog'],  # Keylog
                row['sentiment']  # Sentiment
            ])

        # Append the feature vector to x_data
        x_data[log_type].append(feature_vector)
        # Append the label to y_data
        y_data[log_type].append(row['label'])

def process_date(array):
    temp_dict = {}
    for i in range(len(array)):
        timestamp = pd.Timestamp(float(array[i][0]), unit='s')
        t_date = timestamp.date().strftime('%Y-%m-%d')
        if t_date not in temp_dict:
            temp_dict[t_date] = [i]
        else:
            temp_dict[t_date].append(i)
    return temp_dict

def append_to_chronological_list(data_dict, data_x, log_type, t_date, required_fields):
    entries = []
    for s_date, nodes in data_dict.items():
        if s_date == t_date.strftime('%Y-%m-%d'):
            for node in nodes:
                entry = {
                    'date': t_date,
                    'log_type': log_type,
                    'timestamp': float(data_x[node][0]),
                    'user': int(data_x[node][1]),
                    'pc': int(data_x[node][2]),
                    'node': node
                }
                # Add activity field if required
                if 'activity' in required_fields:
                    entry['activity'] = int(data_x[node][3])
                entries.append(entry)
    return entries

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    sentiment = 0 if (scores['neg'] > scores['neu'] and scores['neg'] > scores['pos']) else 1
    return sentiment

def preprocess_data_month(data: dict, start_date: datetime, end_date: datetime):
    new_data = {}
    x_data = {}
    y_data = {}

    to_remove = ["answers"]
    log_types = [log for log in data.keys() if log not in to_remove]

    month_answers = data["answers"][(data["answers"]['date'] >= start_date) & (data["answers"]['date'] < end_date)].reset_index(drop=True)

    for key in log_types:
        data[key]['date'] = pd.to_datetime(data[key]['date'], format="%m/%d/%Y %H:%M:%S")
        new_data[key] = data[key][(data[key]['date'] >= start_date) & (data[key]['date'] < end_date)].reset_index(drop=True)
        merged_data = new_data[key].merge(month_answers[['id', 'log_type']], on='id', how='left')
        new_data[key]['label'] = (merged_data['log_type'] == key)
        new_data[key]['label'] = new_data[key]['label'].astype(int)
        new_data[key]["hour"] = new_data[key]["date"].dt.hour
        new_data[key]["off_work"] = new_data[key]['hour'].apply(lambda x: 1 if x < 9 or x > 17 else 0)
        x_data[key] = []
        y_data[key] = []

    users = pd.concat([new_data[log_type]["user"] for log_type in new_data]).unique()
    pcs = pd.concat([new_data[log_type]["pc"] for log_type in new_data]).unique()

    new_data["file"]["sentiment"] = new_data["file"]["content"].parallel_apply(get_sentiment)
    new_data["file"]["filetype"] = new_data["file"]["filename"].parallel_apply(lambda x: x.split(".")[-1])
    new_data["http"]["domain"] = new_data["http"]["url"].str.split("/").str[2]
    new_data["http"]["sentiment"] = new_data["http"]["content"].parallel_apply(get_sentiment)
    new_data["http"]["keylog"] = new_data["http"]['content'].str.lower().str.contains('keylog')
    new_data["http"]["keylog"] = new_data["http"]["keylog"].astype(int)

    logon_activity = new_data["logon"]["activity"].unique()
    device_activity = new_data["device"]["activity"].unique()
    filenames = new_data["file"]["filename"].unique()
    filetypes = new_data["file"]["filetype"].unique()
    domains = new_data["http"]["domain"].unique()

    encodings = {
        "type": {key: i for key, i in enumerate(log_types)},
        "user": {user: i for i, user in enumerate(users)},
        "pc": {pc: i for i, pc in enumerate(pcs)},
        "logon_activity": {activity: i for i, activity in enumerate(logon_activity)},
        "device_activity": {activity: i for i, activity in enumerate(device_activity)},
        "filename": {filename: i for i, filename in enumerate(filenames)},
        "filetype": {filetype: i for i, filetype in enumerate(filetypes)},
        "domain": {domain: i for i, domain in enumerate(domains)}
    }

    reverse_encodings = {}
    for key in encodings.keys():
        reverse_encodings[key] = {v: k for k, v in encodings[key].items()}

    for key in log_types:
        process_log_type(key, new_data[key], x_data, y_data, encodings)

    sorted_date_dict = {
        "logon": process_date(x_data["logon"]),
        "device": process_date(x_data["device"]),
        "http": process_date(x_data["http"]),
        "file": process_date(x_data["file"])
    }

    chronological_list = []
    delta = end_date - start_date

    for i in tqdm(range(delta.days), desc="Processing Chronological Data"):
        t_date = start_date + timedelta(days=i)
        for key in sorted_date_dict.keys():
            if t_date.strftime('%Y-%m-%d') in sorted_date_dict[key]:
                chronological_list += append_to_chronological_list(sorted_date_dict[key], x_data[key], key, t_date, required_fields=(['activity'] if (key in ['logon', 'device']) else []))

    chronological_df = pd.DataFrame(chronological_list)

    return {
        "x": x_data,
        "y": y_data,
        "encodings": encodings,
        "reverse_encodings": reverse_encodings,
        "chronological_df": chronological_df
    }

def preprocess_data(data: dict):
    start_date = [datetime(2010, 7, 1, 0, 0, 0), datetime(2010, 8, 1, 0, 0, 0)] # can be single datetime as well
    end_date = [datetime(2010, 8, 1, 0, 0, 0), datetime(2010, 9, 1, 0, 0, 0)] # can be single datetime as well
    if isinstance(start_date, list):
        processed_data = []
        for i in range(len(start_date)):
            new_data = preprocess_data_month(data, start_date[i], end_date[i])
            new_data["start_date"] = start_date[i]
            new_data["end_date"] = end_date[i]
            processed_data.append(new_data)
        return processed_data
    else:
        new_data = preprocess_data_month(data, start_date, end_date)
        new_data["start_date"] = start_date
        new_data["end_date"] = end_date
        return new_data

    