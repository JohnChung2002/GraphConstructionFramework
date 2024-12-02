from .utils import rule_wrapper, all_subsets
from datetime import timedelta
from tqdm import tqdm

def edge_associate(data, j, edges):
    if len(data) < 2:
        return
    data1 = data.iloc[j-1]
    data2 = data.iloc[j]
    edges[f"{data1['log_type']}_associate_{data2['log_type']}"][0].append(data1['node'])
    edges[f"{data1['log_type']}_associate_{data2['log_type']}"][1].append(data2['node'])

def next_edge_associate(today_data, next_day_data, edges):
    if len(today_data) < 2 or len(next_day_data) < 2:
        return
    data1 = today_data.iloc[0]
    data2 = today_data.iloc[-1]
    data3 = next_day_data.iloc[0]
    data4 = next_day_data.iloc[-1]
    edges[f"{data1['log_type']}_associate_{data3['log_type']}"][0].append(data1['node'])
    edges[f"{data1['log_type']}_associate_{data3['log_type']}"][1].append(data3['node'])
    edges[f"{data2['log_type']}_associate_{data4['log_type']}"][0].append(data2['node'])
    edges[f"{data2['log_type']}_associate_{data4['log_type']}"][1].append(data4['node'])

@rule_wrapper({1})
def process_rule_1(today_data, edges, **kwargs):
    for j in range(1, len(today_data)):
        edge_associate(today_data, j, edges)

@rule_wrapper({2, 3})
def process_rule_2_3(today_data, pc_encodings, reverse_logon_activity_encodings, reverse_device_activity_encodings, edges, **kwargs):
    rule_set = kwargs.get("rule_set", [])
    for pc in pc_encodings:
        process_rule_2(today_data, pc, edges, rule_set=rule_set)
        process_rule_3(today_data, pc, reverse_logon_activity_encodings, reverse_device_activity_encodings, edges, rule_set=rule_set)

@rule_wrapper({2})
def process_rule_2(today_data, pc, edges, **kwargs):
    same_pc_data = today_data[today_data["pc"] == pc]
    for j in range(1, len(same_pc_data)):
        edge_associate(same_pc_data, j, edges)

@rule_wrapper({3})
def process_rule_3(today_data, pc, reverse_logon_activity_encodings, reverse_device_activity_encodings, edges, **kwargs):
    same_pc_data = today_data[today_data["pc"] == pc]
    for activity in reverse_logon_activity_encodings:
        logon_activity_df = same_pc_data[((same_pc_data["log_type"] == "logon") & (same_pc_data["activity"] == activity))]
        for j in range(1, len(logon_activity_df)):
            edge_associate(logon_activity_df, j, edges)

    for activity in reverse_device_activity_encodings:
        device_activity_df = same_pc_data[((same_pc_data["log_type"] == "device") & (same_pc_data["activity"] == activity))]
        for j in range(1, len(device_activity_df)):
            edge_associate(device_activity_df, j, edges)

    same_day_http_user_data = same_pc_data[same_pc_data["log_type"] == "http"]
    for j in range(1, len(same_day_http_user_data)):
        edge_associate(same_day_http_user_data, j, edges)

    same_day_file_user_data = same_pc_data[same_pc_data["log_type"] == "file"]
    for j in range(1, len(same_day_file_user_data)):
        edge_associate(same_day_file_user_data, j, edges)

@rule_wrapper({4})
def process_rule_4(today_data, next_day_data, edges, **kwargs):
    next_edge_associate(today_data, next_day_data, edges)

@rule_wrapper({5, 6})
def process_rule_5_6(today_data, next_day_data, pc_encodings, reverse_logon_activity_encodings, reverse_device_activity_encodings, edges, **kwargs):
    rule_set = kwargs.get("rule_set", [])
    for pc in pc_encodings:
        process_rule_5(today_data, next_day_data, pc, edges, rule_set=rule_set)
        process_rule_6(today_data, next_day_data, pc, reverse_logon_activity_encodings, reverse_device_activity_encodings, edges, rule_set=rule_set)

@rule_wrapper({5})
def process_rule_5(today_data, next_day_data, pc, edges, **kwargs):
    today_same_pc_data = today_data[today_data["pc"] == pc]
    next_day_same_pc_data = next_day_data[next_day_data["pc"] == pc]
    next_edge_associate(today_same_pc_data, next_day_same_pc_data, edges)

@rule_wrapper({6})
def process_rule_6(today_data, next_day_data, pc, reverse_logon_activity_encodings, reverse_device_activity_encodings, edges, **kwargs):
    today_same_pc_data = today_data[today_data["pc"] == pc]
    next_day_same_pc_data = next_day_data[next_day_data["pc"] == pc]

    for activity in reverse_logon_activity_encodings:
        logon_activity_df_1 = today_same_pc_data[((today_same_pc_data["log_type"] == "logon") & (today_same_pc_data["activity"] == activity))]
        logon_activity_df_2 = next_day_same_pc_data[((next_day_same_pc_data["log_type"] == "logon") & (next_day_same_pc_data["activity"] == activity))]
        next_edge_associate(logon_activity_df_1, logon_activity_df_2, edges)

    for activity in reverse_device_activity_encodings:
        device_activity_df_1 = today_same_pc_data[((today_same_pc_data["log_type"] == "device") & (today_same_pc_data["activity"] == activity))]
        device_activity_df_2 = next_day_same_pc_data[((next_day_same_pc_data["log_type"] == "device") & (next_day_same_pc_data["activity"] == activity))]
        next_edge_associate(device_activity_df_1, device_activity_df_2, edges)

    today_http_user_data = today_same_pc_data[today_same_pc_data["log_type"] == "http"]
    next_day_http_user_data = next_day_same_pc_data[next_day_same_pc_data["log_type"] == "http"]
    next_edge_associate(today_http_user_data, next_day_http_user_data, edges)

    today_file_user_data = today_same_pc_data[today_same_pc_data["log_type"] == "file"]
    next_day_file_user_data = next_day_same_pc_data[next_day_same_pc_data["log_type"] == "file"]
    next_edge_associate(today_file_user_data, next_day_file_user_data, edges)

def process_day(rule_list, date, chronological_df, pc_encodings, reverse_logon_activity_encodings, reverse_device_activity_encodings, delta):
    # Initialize a local edges dictionary for this day's processing
    edges = {
        "logon_associate_logon" : [[], []],
        "logon_associate_device" : [[], []],
        "logon_associate_http" : [[], []],
        "logon_associate_file" : [[], []],
        "device_associate_logon" : [[], []],
        "device_associate_device" : [[], []],
        "device_associate_http" : [[], []],
        "device_associate_file" : [[], []],
        "http_associate_logon" : [[], []],
        "http_associate_device" : [[], []],
        "http_associate_http" : [[], []],
        "http_associate_file" : [[], []],
        "file_associate_logon" : [[], []],
        "file_associate_device" : [[], []],
        "file_associate_http" : [[], []],
        "file_associate_file" : [[], []]
    }

    today_data = chronological_df[chronological_df["date"].dt.date == date.date()].sort_values(by=['timestamp'])

    # Rule 1 - Chronological Activity
    process_rule_1(today_data, edges, rule_set=rule_list)

    # Rule 2, 3 - Same PC Activity, Same Operation, etc.
    process_rule_2_3(today_data, pc_encodings, reverse_logon_activity_encodings, reverse_device_activity_encodings, edges, rule_set=rule_list)

    # Rule 4, 5, 6 - Next day associations
    if (date + timedelta(days=1)) <= (date + timedelta(days=delta.days)):
        next_day_data = chronological_df[chronological_df["date"].dt.date == (date + timedelta(days=1)).date()].sort_values(by=['timestamp'])
        process_rule_4(today_data, next_day_data, edges, rule_set=rule_list)

        process_rule_5_6(today_data, next_day_data, pc_encodings, reverse_logon_activity_encodings, reverse_device_activity_encodings, edges, rule_set=rule_list)

    return edges

def merge_edges(edges_list):
    merged_edges = {
        "logon_associate_logon" : [[], []],
        "logon_associate_device" : [[], []],
        "logon_associate_http" : [[], []],
        "logon_associate_file" : [[], []],
        "device_associate_logon" : [[], []],
        "device_associate_device" : [[], []],
        "device_associate_http" : [[], []],
        "device_associate_file" : [[], []],
        "http_associate_logon" : [[], []],
        "http_associate_device" : [[], []],
        "http_associate_http" : [[], []],
        "http_associate_file" : [[], []],
        "file_associate_logon" : [[], []],
        "file_associate_device" : [[], []],
        "file_associate_http" : [[], []],
        "file_associate_file" : [[], []]
    }

    for edges in edges_list:
        for key in merged_edges.keys():
            merged_edges[key][0].extend(edges[key][0])
            merged_edges[key][1].extend(edges[key][1])

    return merged_edges

def process_rules_each(data, rule_list):
    results = []

    chronological_df = data["chronological_df"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    delta = end_date - start_date
    pc_encodings = data["encodings"]["pc"]
    reverse_logon_activity_encodings = data["reverse_encodings"]["logon_activity"]
    reverse_device_activity_encodings = data["reverse_encodings"]["device_activity"]

    for i in tqdm(range(delta.days), desc="Processing Days for Rules"):
        date = start_date + timedelta(days=i)
        out = process_day(
            rule_list,
            date,
            chronological_df,
            pc_encodings,
            reverse_logon_activity_encodings,
            reverse_device_activity_encodings,
            delta
        )
        results.append(out)

    # Merge edges from all days
    edges = merge_edges(results)
    return edges

def process_rules(data):
    rule_list = [i for i in range(1, 7)]
    
    subsets = all_subsets(rule_list)
    print(f"Total number of unique subsets: {len(subsets)}")

    if isinstance(data, list):
        combination_list = []
        for d in data:
            combinations = {}
            for subset in subsets:
                temp = [str(i) for i in subset]
                combinations["+".join(temp)] = process_rules_each(d, subset)
            combination_list.append(combinations)
        return combination_list
    else:
        combinations = {}
        for subset in subsets:
            temp = [str(i) for i in subset]
            combinations["+".join(temp)] = process_rules_each(data, subset)
        return combinations
