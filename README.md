Self-Healing Graph Neural Network for Supply Chain Anomaly Detection
A PyTorch Geometric-based framework for anomaly detection and autonomous self-healing in supply chain logistics graphs. This system constructs similarity graphs from structured order data, applies advanced GNN models (GCN, GAT, GraphSAGE, GIN), detects compromised nodes, and automatically "heals" the graph by removing detected anomalies.

🚀 Features
Dynamic Graph Construction: Builds k-NN similarity graphs from supply chain logistics records.

Imbalance Handling: Applies SMOTE to address class imbalance between healthy and compromised nodes.

Multi-Model Support: Supports GCN, GAT, GraphSAGE, and GIN binary classifiers (Healthy vs. Compromised).

Self-Healing Logic: Automatically removes predicted compromised nodes, enabling self-healing supply chain graphs.

Comprehensive Evaluation: Computes precision, recall, F1, specificity, accuracy, confusion matrices, and healing audit tables.

Rich Visualizations: Provides training curves (loss, accuracy, F1), healing impact visualizations, confusion matrices, and graph snapshots before/after healing.

📁 Project Structure
css
Copy
Edit
├── main.py
├── requirements.txt
├── LICENSE
├── README.md
├── utils/
│   ├── load_data.py
│   └── graph_utils.py
├── model/
│   └── gcn.py
├── training/
│   └── train_eval.py
├── healing/
│   ├── graph_healing.py
│   └── visualization.py
📝 Outputs
Classification Reports: Precision, recall, F1, and specificity before and after healing.

Healing Effect Summaries: Audit tables showing node status transitions (TP, FP, TN, FN).

Graph Visualizations: Node classification results and healed graph structure.

Confusion Matrices: For each GNN model.

Training Curves: Loss, accuracy, F1 progression, and more.

Get Started:
See main.py for an end-to-end demonstration and experiment script.
Check requirements.txt for dependencies and installation instructions.

✍️ Authors
Developed by Dr. Debashis Das
Postdoctoral Researcher | Meharry School of Applied Computational Sciences
