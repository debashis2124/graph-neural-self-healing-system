# Self-Healing Graph Neural Network

A Graph Neural Network (GNN)-based anomaly detection and self-healing system for supply chain logistics. This framework builds similarity graphs from order data, detects anomalies using GCN models, and automatically "heals" the graph by removing predicted compromised nodes.

---

## 🚀 Features

- Constructs dynamic graphs from structured logistics data
- Applies SMOTE for class imbalance handling
- Trains a GCN-based binary classifier (Healthy vs. Compromised)
- Performs self-healing by removing compromised nodes
- Evaluates healing performance with precision, recall, F1, and specificity
- Provides visualizations for model metrics and graph healing impact

---

## 📁 Project Structure

├── main.py
├── requirements.txt
├── LICENSE
├── README.md
├── utils/
│ ├── load_data.py
│ └── graph_utils.py
├── model/
│ └── gcn.py
├── training/
│ └── train_eval.py
├── healing/
│ ├── graph_healing.py
│ └── visualization.py


## Output

Classification report before healing
Healing effect summary
Graph visualizations before and after healing
Confusion matrix and audit tables
Training curves (loss, accuracy, F1)

✍️ Authors
Developed by Dr. Debashis Das
Postdoctoral Researcher | Meharry School of Applied Computational Sciences
