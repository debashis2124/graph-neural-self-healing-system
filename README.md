Self-Healing GNN for Supply Chain
This project uses graph neural networks (GNNs) to find and fix (self-heal) problems in supply chain data. It uses order data to build graphs, detects "compromised" (anomalous) nodes, and removes them to improve the graph.

🚀 What it Does
Makes graphs from logistics (order) data

Fixes class imbalance using SMOTE

Trains different GNN models (GCN, GAT, GraphSAGE, GIN)

Finds compromised nodes (bad/at-risk orders)

Removes compromised nodes to heal the graph

Shows results: accuracy, F1 score, recall, precision, specificity

Draws plots for metrics and healing effect

📂 Project Files
bash
Copy
Edit
main.py               # Main script to run everything
requirements.txt      # List of packages to install
LICENSE
README.md
utils/
    load_data.py      # For reading and preparing data
    graph_utils.py    # Helpers for graph operations
model/
    gcn.py            # GCN model code
training/
    train_eval.py     # Training and evaluation functions
healing/
    graph_healing.py  # Healing (removal) functions
    visualization.py  # Plots and graph visuals
📊 What You Get
Classification reports (before/after healing)

Healing effect summary table

Graph plots (before and after healing)

Confusion matrices for each GNN

Training curves (loss, accuracy, F1 score)

To start:
Check main.py to run an experiment.
See requirements.txt for what to install.

✍️ Authors
Developed by Dr. Debashis Das
Postdoctoral Researcher | Meharry School of Applied Computational Sciences
