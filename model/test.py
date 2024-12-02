import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, SAGEConv
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# === 1. Node Encoder ===
class NodeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.encoder(x)

# === 2. Heterogeneous Graph Neural Network ===
class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_dim):
        super().__init__()
        self.convs = nn.ModuleList([
            HeteroConv(
                {edge_type: SAGEConv((-1, -1), hidden_dim)
                 for edge_type in metadata['edge_types']},
                aggr='mean'
            )
            for _ in range(2)  # 2 layers of message passing
        ])
        self.node_encoders = nn.ModuleDict({
            node_type: NodeEncoder(metadata['dim'], hidden_dim)
            for node_type in metadata['node_types']
        })

    def forward(self, x_dict, edge_index_dict):
        x_dict = {node_type: self.node_encoders[node_type](x)
                  for node_type, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return x_dict

# === 3. Classification Head ===
class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.classifier(x)

# === 4. Preprocessing: Normalize Features ===
def normalize_features(graph):
    for node_type in graph.node_types:
        sk_scaler = StandardScaler()
        graph[node_type]['x'] = torch.tensor(
            sk_scaler.fit_transform(graph[node_type]['x'].cpu().numpy()), dtype=torch.float32
        )
    return graph

# === 5. SMOTE for Balancing Classes ===
def apply_smote(features, labels):
    unique_classes, class_counts = torch.unique(labels, return_counts=True)
    if len(unique_classes) <= 1 or class_counts.min().item() < 6:  # Ensure at least 6 samples for SMOTE
        print(f"Skipping SMOTE for node type due to insufficient samples: {class_counts.tolist()}")
        return features, labels

    smote = SMOTE(k_neighbors=min(5, class_counts.min().item() - 1))  # Adjust k_neighbors dynamically
    features_resampled, labels_resampled = smote.fit_resample(features.cpu().numpy(), labels.cpu().numpy())
    return torch.tensor(features_resampled, dtype=torch.float32), torch.tensor(labels_resampled, dtype=torch.long)

def balance_classes_with_smote(graph):
    for node_type in graph.node_types:
        graph[node_type]['x'], graph[node_type]['y'] = apply_smote(
            graph[node_type]['x'], graph[node_type]['y']
        )
    return graph

# === 6. Weighted Loss ===
def compute_class_weights(y):
    class_counts = torch.bincount(y)
    total_samples = len(y)
    return total_samples / (len(class_counts) * class_counts)

def dynamic_weighted_loss(logits, targets, weights, epoch, total_epochs):
    weight_scale = torch.linspace(1.0, 2.0, total_epochs)[epoch]
    return nn.CrossEntropyLoss(weight=(weights * weight_scale).to(targets.device))(logits, targets)

# === 7. Training Loop ===
def train_model(model, classifiers, graph, optimizer, scheduler, epochs, rule_comb, results):
    results_text = []
    
    metadata = {
        "node_types": graph.metadata()[0],
        "edge_types": graph.metadata()[1]
    }
    weights = {
        node_type: compute_class_weights(graph[node_type]['y']).to(torch.float32)
        for node_type in graph.node_types
    }

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_dict = {node_type: graph[node_type]['x'] for node_type in graph.node_types}
        edge_index_dict = {edge_type: torch.tensor(graph[edge_type]['edge_index'], dtype=torch.int64) for edge_type in graph.edge_types}

        out_dict = model(x_dict, edge_index_dict)
        total_loss = 0

        for node_type, out in out_dict.items():
            logits = classifiers[node_type](out)
            targets = graph[node_type]['y']
            total_loss += dynamic_weighted_loss(logits, targets, weights[node_type], epoch, epochs)

            # Training Classification Report
            targets_cpu = targets.cpu().numpy()
            preds_cpu = logits.argmax(dim=1).cpu().numpy()
            report = classification_report(targets_cpu, preds_cpu, zero_division=0)
            text = f"\n{node_type} Training Report (Epoch {epoch + 1}):\n{report}"
            results_text.append(text)
            print(text)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}")
    
    with open(f"{results}/{rule_comb}_training_{node_type}", 'w') as f:
        f.write('\n'.join(results_text))

# === 8. Validation/Test Evaluation ===
def evaluate_model(model, classifiers, graph, rule_comb, results):
    results_text = []
    
    model.eval()
    with torch.no_grad():
        x_dict = {node_type: graph[node_type]['x'] for node_type in graph.node_types}
        edge_index_dict = {edge_type: torch.tensor(graph[edge_type]['edge_index'], dtype=torch.int64) for edge_type in graph.edge_types}

        out_dict = model(x_dict, edge_index_dict)

        for node_type, out in out_dict.items():
            logits = classifiers[node_type](out)
            targets = graph[node_type]['y']
            preds = logits.argmax(dim=1)

            # Classification Report
            report = classification_report(
                targets.cpu().numpy(),
                preds.cpu().numpy(),
                target_names=[str(i) for i in range(2)],
                zero_division=0
            )
            
            text = f"\n{node_type} Validation/Test Report:\n{report}"
            results_text.append(text)
            print(text)

            # ROC-AUC
            probs = logits.softmax(dim=1)[:, 1].cpu().numpy()
            roc_auc = roc_auc_score(targets.cpu().numpy(), probs)
            print(f"{node_type} ROC-AUC: {roc_auc:.4f}")

    with open(f"{results}/{rule_comb}_validation_{node_type}.txt", 'w') as f:
        f.write('\n'.join(results_text))

# === 9. Main Execution ===
def train_evaluate_model(graphs: list, edges, results: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edges_namelist = []
    if isinstance(edges, list):
        for i in range(len(edges)):
            edges_namelist.extend([f"Graph{i}_{key}" for key in edges[i].keys()])
    else:
        edges_namelist.extend([key for key in edges.keys()])
    epochs = 20
    for i in range(0, len(graphs), 2):
        dataset = normalize_features(graphs[i])
        dataset = balance_classes_with_smote(dataset)

        dataset.to(device)

        rule_comb = edges_namelist[(int(i/2))]

        validation_graph = graphs[i+1].to(device)

        # Get the maximum number of features across all node types
        desired_dim = 0
        for node_type in dataset.node_types:
            desired_dim = max(desired_dim, dataset[node_type].x.shape[1])

        for node_type in dataset.node_types:
            current_dim = dataset[node_type].x.shape[1]
            if current_dim < desired_dim:
                padding = torch.zeros((dataset[node_type].x.shape[0], desired_dim - current_dim), device=dataset[node_type].x.device)
                dataset[node_type].x = torch.cat([dataset[node_type].x, padding], dim=1)
            elif current_dim > desired_dim:
                dataset[node_type].x = dataset[node_type].x[:, :desired_dim]
        
        for node_type in validation_graph.node_types:
            current_dim = validation_graph[node_type].x.shape[1]
            if current_dim < desired_dim:
                padding = torch.zeros((validation_graph[node_type].x.shape[0], desired_dim - current_dim), device=validation_graph[node_type].x.device)
                validation_graph[node_type].x = torch.cat([validation_graph[node_type].x, padding], dim=1)
            elif current_dim > desired_dim:
                validation_graph[node_type].x = validation_graph[node_type].x[:, :desired_dim]

        # Metadata
        metadata = {
            "dim": desired_dim,
            "node_types": dataset.metadata()[0],
            "edge_types": dataset.metadata()[1],
        }

        # Initialize Model and Components
        hidden_dim = 64
        model = HeteroGNN(metadata, hidden_dim).to(device)
        classifiers = nn.ModuleDict({
            node_type: ClassificationHead(hidden_dim).to(device)
            for node_type in metadata['node_types']
        })
        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # Train the Model
        train_model(model, classifiers, dataset, optimizer, scheduler, epochs, rule_comb, results)

        # Evaluate on Validation/Test Graph
        evaluate_model(model, classifiers, validation_graph, rule_comb, results)