import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops

def build_graph(processed_data: list | dict, edges: list | dict):
    graphs = []
    if isinstance(processed_data, list):
        for i, data in enumerate(processed_data):
            for subset in edges[i].keys():
                graph = HeteroData()
                for key in data["x"].keys():
                    graph[key].x = torch.tensor(data["x"][key])
                    graph[key].y = torch.tensor(data["y"][key], dtype=torch.long)
                    graph[key].num_nodes = len(data["x"][key])
                for edge_key in edges[i][subset].keys():
                    node1, relationship, node2 = edge_key.split("_")
                    edges[i][subset][edge_key] = torch.tensor(edges[i][subset][edge_key])
                    graph[node1, relationship, node2].edge_index = edges[i][subset][edge_key]
                for key in graph.keys():
                    if 'edge_index' in graph[key]:
                        graph[key].edge_index = add_self_loops(graph[key].edge_index, num_nodes=graph[key].num_nodes)[0]
                del graph['edge_index']
                del graph['num_nodes']
                del graph['x']
                del graph['y']
                graphs.append(graph)
    else:
        for subset in edges.keys():
            graph = HeteroData()
            for key in processed_data["x"].keys():
                graph[key].x = torch.tensor(processed_data["x"][key])
                graph[key].y = torch.tensor(processed_data["y"][key], dtype=torch.long)
                graph[key].num_nodes = len(processed_data["x"][key])
            for edge_key in edges[subset].keys():
                node1, relationship, node2 = edge_key.split("_")
                edges[subset][key] = torch.tensor(edges[subset][key])
                graph[node1, relationship, node2].edge_index = edges[subset][key]  
            for key in graph.keys():
                if 'edge_index' in graph[key]:
                    graph[key].edge_index = add_self_loops(graph[key].edge_index, num_nodes=graph[key].num_nodes)[0]
            del graph['edge_index']
            del graph['num_nodes']
            del graph['x']
            del graph['y']
            graphs.append(graph)
    return graphs