# Self-Healing Graph Neural Network for Supply Chain Anomaly Detection

This project implements a Graph Neural Network (GNN)-based anomaly detection and self-healing system for supply chain logistics data. The framework constructs similarity graphs from order records, detects anomalous (compromised) nodes using several GNN models, and applies self-healing by removing these compromised nodes from the graph.

---

## 🚀 Features

- Dynamic similarity graph construction from logistics/order data
- SMOTE-based class imbalance handling
- Multiple GNN models: GCN, GAT, GraphSAGE, GIN
- Binary classification: Healthy vs. Compromised nodes
- Automated graph healing by node removal
- Performance metrics: Accuracy, Precision, Recall, F1, Specificity
- Training curves, confusion matrices, and healing visualizations

---

## 🛠️ Installation

Clone the repo and install dependencies:
git clone https://github.com/your-username/self-healing-gnn.git
cd self-healing-gnn
pip install -r requirements.txt

## Outputs
Pre- and post-healing classification reports
Healing effect summaries and audit tables
Confusion matrices (per GNN model)
Plots: loss, accuracy, F1, precision, recall, specificity, and healing ratio
Graph visualizations: before and after self-healing

✍️ Authors
Developed by Dr. Debashis Das
Postdoctoral Researcher | Meharry School of Applied Computational Sciences
