#Different models Comparision

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from collections import Counter
from tabulate import tabulate

# --------- Configs ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
LEARNING_RATE = 0.002
HIDDEN_CHANNELS = 32
K_NEIGHBORS = 10
SMOTE_RATIO = 0.5
SIMILARITY_THRESHOLD = 0.8
DISTANCE_THRESHOLD = 1 - SIMILARITY_THRESHOLD

# --------- Load Data ----------
df = pd.read_excel('Supply chain logisitcs problem.xlsx', sheet_name='OrderList')
df['Label'] = df['Ship Late Day count'].apply(lambda x: 1 if x > 3 else 0)
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Hour'] = df['Order Date'].dt.hour
df['Day'] = df['Order Date'].dt.day
df['Weekday'] = df['Order Date'].dt.weekday
df.ffill(inplace=True)

categorical_cols = ['Origin Port', 'Carrier', 'Service Level', 'Customer', 'Product ID', 'Plant Code', 'Destination Port']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

selected_features = [
    'Order ID', 'Origin Port', 'Carrier', 'TPT', 'Service Level',
    'Ship ahead day count', 'Ship Late Day count', 'Customer',
    'Product ID', 'Plant Code', 'Destination Port',
    'Unit quantity', 'Weight', 'Hour', 'Day', 'Weekday'
]
scaler = StandardScaler()
features = scaler.fit_transform(df[selected_features])
labels = df['Label']

# --------- Balance with SMOTE ----------
data_counts = Counter(labels)
minority_class = min(data_counts, key=data_counts.get)
k_neighbors = min(5, data_counts[minority_class] - 1) if data_counts[minority_class] > 1 else 1
smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=42, k_neighbors=k_neighbors)
features_resampled, labels_resampled = smote.fit_resample(features, labels)

# --------- Build Graph (KNN) ----------
G = nx.Graph()
for i in range(len(features_resampled)):
    G.add_node(i, x=torch.tensor(features_resampled[i], dtype=torch.float32), y=int(labels_resampled[i]))
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine')
knn.fit(features_resampled)
distances, indices = knn.kneighbors(features_resampled)
for i, (dists, neighbors) in enumerate(zip(distances, indices)):
    for j, dist in zip(neighbors[1:], dists[1:]):
        if dist < DISTANCE_THRESHOLD:
            G.add_edge(i, j, weight=1 - dist)

# --------- PyG Data Object ----------
data = Data()
data.edge_index = torch.tensor(list(G.edges)).t().contiguous().to(device)
data.edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges()], dtype=torch.float32).to(device)
data.x = torch.stack([G.nodes[i]['x'] for i in G.nodes()]).to(device)
data.y = torch.tensor([G.nodes[i]['y'] for i in G.nodes()], dtype=torch.long).to(device)
data.num_node_features = data.x.shape[1]

indices = np.arange(data.num_nodes)
y_np = data.y.cpu().numpy()
train_idx, test_idx = train_test_split(indices, stratify=y_np, test_size=0.3, random_state=42)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
data.train_mask[train_idx] = True
data.test_mask[test_idx] = True

# --------- Model Definitions ----------
class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)
        self.dropout = nn.Dropout(0.5)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=2):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, 2, heads=1)
        self.dropout = nn.Dropout(0.5)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class SAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 2)
        self.dropout = nn.Dropout(0.5)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GINNet(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        from torch_geometric.nn import GINConv
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_channels, 2))
        self.conv2 = GINConv(nn2)
        self.dropout = nn.Dropout(0.5)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# --------- Unified Train/Eval Loop ----------
def train_gnn_epochwise(model_class, data, name, hidden_channels, epochs=50):
    model = model_class(data.num_node_features, hidden_channels).to(device)
    label_tensor = data.y[data.train_mask]
    class_weights = 1.0 / torch.bincount(label_tensor).float()
    class_weights /= class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    train_loss, accs, f1s, precs, recs, specs = [], [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            _, pred = out.max(dim=1)
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()
            accs.append(accuracy_score(y_true, y_pred))
            precs.append(precision_score(y_true, y_pred, zero_division=0))
            recs.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
            cm = confusion_matrix(y_true, y_pred)
            specs.append(cm[0,0]/(cm[0,0]+cm[0,1]+1e-6))
    return {
        "name": name,
        "train_loss": train_loss,
        "accuracy": accs,
        "precision": precs,
        "recall": recs,
        "f1": f1s,
        "specificity": specs
    }

# --------- Train All Models ---------
all_models = [
    ("GCN", GCNNet),
    ("GAT", GATNet),
    ("GraphSAGE", SAGENet),
    ("GIN", GINNet)
]

results = []
for name, model_class in all_models:
    print(f"Training {name} ...")
    curves = train_gnn_epochwise(model_class, data, name, HIDDEN_CHANNELS, EPOCHS)
    results.append(curves)

# --------- Plot Metric Comparisons (All Models) ---------
def plot_metric_comparison(results, metric, ylabel):
    plt.figure(figsize=(9, 5))
    # List of distinct markers (extend if you have more than 6 models)
    markers = ['o', 's', '^', 'x', 'D', 'P', '*', '<', '>', 'v', 'h', '+', '1', '2', '3', '4']
    for idx, res in enumerate(results):
        marker = markers[idx % len(markers)]
        plt.plot(range(1, len(res[metric]) + 1), res[metric], label=res["name"], marker=marker, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison Among Various GNN Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_metric_comparison(results, "accuracy", "Accuracy")
plot_metric_comparison(results, "precision", "Precision")
plot_metric_comparison(results, "recall", "Recall")
plot_metric_comparison(results, "f1", "F1 Score")
plot_metric_comparison(results, "specificity", "Specificity")

# --------- Final Metric Table (Last Epoch) ---------
summary = []
for r in results:
    summary.append([
        r["name"],
        f"{r['accuracy'][-1]:.4f}",
        f"{r['precision'][-1]:.4f}",
        f"{r['recall'][-1]:.4f}",
        f"{r['f1'][-1]:.4f}",
        f"{r['specificity'][-1]:.4f}"
    ])
print("\nFinal Model Metrics (last epoch):")
print(tabulate(summary, headers=["Model","Accuracy","Precision","Recall","F1","Specificity"], tablefmt="fancy_grid"))

# You can now also apply your healing logic per model if you want further comparison after healing!
# ---------- Healing logic function -------------
def evaluate_healing(model_class, data, hidden_channels, name):
    # Re-train (or load) the best model for healing evaluation
    model = model_class(data.num_node_features, hidden_channels).to(device)
    label_tensor = data.y[data.train_mask]
    class_weights = 1.0 / torch.bincount(label_tensor).float()
    class_weights /= class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    best_f1 = 0

    # --- Train for healing evaluation ---
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, pred = out.max(dim=1)
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_out = out
                best_pred = pred

    # -------- Healing process ----------
    all_pred = best_pred.cpu().numpy()
    true_labels = data.y.cpu().numpy()
    total_nodes = len(true_labels)

    # Remove compromised nodes (pred==1)
    healed_G = G.copy()
    for i, pred_label in enumerate(all_pred):
        if pred_label == 1 and i in healed_G:
            healed_G.remove_node(i)
    healed_nodes = list(healed_G.nodes())
    kept_mask = np.zeros(total_nodes, dtype=bool)
    kept_mask[healed_nodes] = True
    removed_mask = ~kept_mask

    # Healing metrics
    true_healthy = (true_labels == 0)
    true_compromised = (true_labels == 1)
    TP = np.sum(removed_mask & true_compromised)
    FP = np.sum(removed_mask & true_healthy)
    TN = np.sum(kept_mask & true_healthy)
    FN = np.sum(kept_mask & true_compromised)
    total = TP + FP + TN + FN

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        "name": name,
        "healing_accuracy": accuracy,
        "healing_precision": precision,
        "healing_recall": recall,
        "healing_f1": f1,
        "healing_specificity": specificity,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN
    }


# ----------- Apply healing to all models --------------
healing_results = []
for name, model_class in all_models:
    print(f"Evaluating healing for {name} ...")
    res = evaluate_healing(model_class, data, HIDDEN_CHANNELS, name)
    healing_results.append(res)

# ---------- Tabular healing summary -----------
healing_summary = [
    [r["name"], f"{r['healing_accuracy']:.4f}", f"{r['healing_precision']:.4f}", f"{r['healing_recall']:.4f}",
     f"{r['healing_f1']:.4f}", f"{r['healing_specificity']:.4f}", r["TP"], r["FP"], r["TN"], r["FN"]]
    for r in healing_results
]
print("\n=== Healing Performance for All Models ===")
print(tabulate(
    healing_summary,
    headers=["Model", "Heal_Acc", "Heal_Prec", "Heal_Recall", "Heal_F1", "Heal_Spec", "TP", "FP", "TN", "FN"],
    tablefmt="fancy_grid"
))

# --------------- Bar plots for Healing ---------------
import matplotlib.pyplot as plt

model_names = [r["name"] for r in healing_results]
metrics = ["healing_accuracy", "healing_precision", "healing_recall", "healing_f1", "healing_specificity"]
labels = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity"]

for metric, label in zip(metrics, labels):
    values = [r[metric] for r in healing_results]
    plt.figure(figsize=(7,4))
    plt.bar(model_names, values, color='dodgerblue', alpha=0.7)
    plt.ylabel(label)
    plt.title(f'Healing {label} Comparison')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Suppose all_models is a list of (model_name, model_instance) tuples, and models are trained.
# E.g. all_models = [("GCN", gcn_model), ("GAT", gat_model), ...]

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# 1. Define count_compromised_nodes function
def count_compromised_nodes(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        _, all_pred = out.max(dim=1)
        all_pred = all_pred.cpu().numpy()
    return np.sum(all_pred == 1), all_pred

# 2. Train each model and collect trained instances
trained_models = []  # (name, trained_model_instance) pairs

for name, ModelClass in [
    ("GCN", GCNNet),
    ("GAT", GATNet),
    ("GraphSAGE", SAGENet),
    ("GIN", GINNet)
]:
    print(f"Training {name} ...")
    model = ModelClass(data.num_node_features, HIDDEN_CHANNELS).to(device)
    label_tensor = data.y[data.train_mask]
    class_weights = 1.0 / torch.bincount(label_tensor).float()
    class_weights /= class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # Simple training loop
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate to get best model (optional)
        model.eval()
        with torch.no_grad():
            _, pred = out.max(dim=1)
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict()

    # Load best model state (optional but good practice)
    model.load_state_dict(best_model_state)
    trained_models.append((name, model))

# 3. Use trained model instances for compromised node counts
compromised_counts = []
compromised_preds = {}

for model_name, model in trained_models:
    count, preds = count_compromised_nodes(model, data)
    compromised_counts.append((model_name, count))
    compromised_preds[model_name] = preds

print("\n====== Compromised Node Count Per Model ======")
print(tabulate(compromised_counts, headers=["Model", "Predicted Compromised"], tablefmt="fancy_grid"))

# 4. Bar plot
model_names = [name for name, _ in compromised_counts]
compromised_vals = [val for _, val in compromised_counts]

plt.figure(figsize=(7,4))
bars = plt.bar(model_names, compromised_vals, color='orange', alpha=0.7)
plt.ylabel("Predicted Compromised Nodes")
plt.title("Compromised Node Count by Model")
for i, v in enumerate(compromised_vals):
    plt.text(i, v + 2, str(v), ha='center', fontsize=12)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def count_nodes_by_class(model, data, true_labels):
    model.eval()
    with torch.no_grad():
        out = model(data)
        _, all_pred = out.max(dim=1)
        all_pred = all_pred.cpu().numpy()
    # Compromised: predicted as compromised (pred == 1)
    compromised = np.sum(all_pred == 1)
    # Recovered: true positive (true_label==1 and pred==1)
    recovered = np.sum((all_pred == 1) & (true_labels == 1))
    return compromised, recovered, all_pred


# Train your models as in the previous example,
# and collect in trained_models = [("GCN", gcn_model), ...] as shown previously

true_labels = data.y.cpu().numpy()
compromised_counts = []
recovered_counts = []
all_preds_dict = {}

for model_name, model in trained_models:
    compromised, recovered, preds = count_nodes_by_class(model, data, true_labels)
    compromised_counts.append((model_name, compromised))
    recovered_counts.append((model_name, recovered))
    all_preds_dict[model_name] = preds

# Table print
print("\n====== Node Count Per Model ======")
table = []
for i in range(len(compromised_counts)):
    table.append([compromised_counts[i][0], compromised_counts[i][1], recovered_counts[i][1]])
print(tabulate(table, headers=["Model", "Compromised Nodes", "Recovered (True Positives)"], tablefmt="fancy_grid"))

# Bar Plot: Compromised vs. Recovered
model_names = [x[0] for x in compromised_counts]
compromised_vals = [x[1] for x in compromised_counts]
recovered_vals = [x[1] for x in recovered_counts]

bar_width = 0.35
x = np.arange(len(model_names))

plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, compromised_vals, width=bar_width, color='orange', label='Compromised')
plt.bar(x + bar_width/2, recovered_vals, width=bar_width, color='seagreen', label='Recovered (TP)')
plt.ylabel("Node Count")
plt.title("Compromised vs. Recovered (True Positives) Nodes by Model")
plt.xticks(x, model_names)
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # If you have all_preds_dict (from previous steps), use that. If not, use this code to get predictions:
# def get_model_preds(model, data):
#     model.eval()
#     with torch.no_grad():
#         out = model(data)
#         _, preds = out.max(dim=1)
#         return preds.cpu().numpy()

# If not already, make sure you have:
# trained_models = [("GCN", gcn_model), ("GAT", gat_model), ("GraphSAGE", sage_model), ("GIN", gin_model)]
# true_labels = data.y.cpu().numpy()

for model_name, model in trained_models:
    preds = get_model_preds(model, data)
    cm = confusion_matrix(true_labels, preds)
    print(f"\n{model_name} Confusion Matrix:\n{cm}\n")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Compromised"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()
