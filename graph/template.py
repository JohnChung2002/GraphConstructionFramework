from torch_geometric.data import Data, HeteroData

def build_graph(processed_data, edges):
    graphs = []

    # If the graph is homogeneous, use the Data class
    # If the graph is heterogeneous, use the HeteroData class
    # Append as many graphs as required to the graphs list

    return graphs