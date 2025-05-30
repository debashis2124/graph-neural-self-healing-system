from utils.load_data import load_and_preprocess_data
from utils.graph_utils import build_graph, convert_to_pyg_data
from model.gcn import GNNAnomalyDetector
from training.train_eval import train_model, evaluate_model
from healing.graph_healing import apply_graph_healing, evaluate_healing_effect
from healing.visualization import (
    plot_training_curves, plot_healing_effect,
    plot_classification_bar, plot_healing_confusion,
    plot_graphs
)

import torch
import numpy as np

# Configuration
FILEPATH = 'Supply chain logisitcs problem.xlsx'
EPOCHS = 50
K_NEIGHBORS = 10
SIMILARITY_THRESHOLD = 0.8

# Step 1: Load data
features, labels = load_and_preprocess_data(FILEPATH)

# Step 2: Graph Construction
G = build_graph(features, labels, k_neighbors=K_NEIGHBORS, sim_thresh=SIMILARITY_THRESHOLD)
data = convert_to_pyg_data(G)

# Step 3: Train GCN model
model, all_pred, y_np, losses, accs, f1s = train_model(data, epochs=EPOCHS)

# Step 4: Evaluate baseline classification
y_true = data.y[data.test_mask].cpu().numpy()
y_pred = all_pred[data.test_mask]
evaluate_model(y_true, y_pred)

# Step 5: Visualize training performance
plot_training_curves(losses, accs, f1s, [1]*EPOCHS, EPOCHS)  # Epoch time optional if collected

# Step 6: Apply healing logic
healed_G = apply_graph_healing(G, data, all_pred)

# Step 7: Healing Evaluation
evaluate_healing_effect(data, healed_G, all_pred)

# Step 8: Graph & Healing Visualizations
before_healthy = np.sum(all_pred == 0)
before_compromised = np.sum(all_pred == 1)

remaining_indices = list(healed_G.nodes())
pred_remaining = all_pred[remaining_indices]
after_healthy = np.sum(pred_remaining == 0)
after_compromised = 0

plot_healing_effect(before_healthy, before_compromised, after_healthy, after_compromised)

# Step 9: Confusion Matrix for Healing
true_labels = data.y.cpu().numpy()
removed_mask = np.ones_like(true_labels, dtype=bool)
removed_mask[remaining_indices] = False
plot_healing_confusion(true_labels, removed_mask)

# Step 10: Breakdown Chart & Graph Drawing
TP = np.sum(removed_mask & (true_labels == 1))
FP = np.sum(removed_mask & (true_labels == 0))
TN = np.sum(~removed_mask & (true_labels == 0))
FN = np.sum(~removed_mask & (true_labels == 1))

plot_classification_bar(TN, FP, TP, FN)
plot_graphs(G, healed_G, y_np)
