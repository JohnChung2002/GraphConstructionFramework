# take in parameters to run
"""
data_loader: path to the data loader (should be in the load_data folder)
data_preprocessor: path to the data preprocessor (should be in the preprocess_data folder)
edge_builder: path to the graph builder (should be in the rules folder)
graph_builder: path to the graph builder (should be in the graph folder)
model: path to the model (should be in the model folder)
results: path to the results 
"""

import importlib
from datetime import datetime

def evaluator(data_loader: str, data_loader_args: dict, data_preprocessor: str, edge_builder: str, graph_builder: str, model: str, results: str):
    initial_time = datetime.now()

    # Load the data loader
    data_loader = importlib.import_module(f"load_data.{data_loader}")

    # Load the data
    start_time = datetime.now()
    print(f"Loading data... | Time: {start_time}")
    data = data_loader.load_data(data_loader_args)
    print(f"Data loaded | Time: {datetime.now() - start_time}")

    # Load the data preprocessor
    data_preprocessor = importlib.import_module(f"preprocess_data.{data_preprocessor}")

    # Preprocess the data
    start_time = datetime.now()
    print(f"Preprocessing data... | Time: {start_time}")
    processed_data = data_preprocessor.preprocess_data(data)
    print(f"Data preprocessed | Time: {datetime.now() - start_time}")

    # Load the graph builder
    edge_builder = importlib.import_module(f"rules.{edge_builder}")
    start_time = datetime.now()
    print(f"Building edges... | Time: {start_time}")
    edges = edge_builder.process_rules(processed_data)
    print(f"Edges built | Time: {datetime.now() - start_time}")

    # Load the graph builder
    graph_builder = importlib.import_module(f"graph.{graph_builder}")
    start_time = datetime.now()
    print(f"Building graph... | Time: {start_time}")
    graph = graph_builder.build_graph(processed_data, edges)
    print(f"Graph built | Time: {datetime.now() - start_time}")

    # Load the model
    model = importlib.import_module(f"model.{model}")
    start_time = datetime.now()
    print(f"Training and evaluating model... | Time: {start_time}")
    model.train_evaluate_model(graph, edges, results)
    print(f"Model trained and evaluated | Time: {datetime.now() - start_time}")
    
    print(f"Evaluation completed | Total Time Taken: {datetime.now() - initial_time}")




