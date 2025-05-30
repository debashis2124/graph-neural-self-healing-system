import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from model.gcn import GNNAnomalyDetector

def train_model(data, epochs=50, lr=0.002):
    device = data.x.device
    indices = np.arange(data.num_nodes)
    y_np = data.y.cpu().numpy()
    train_idx, test_idx = train_test_split(indices, stratify=y_np, test_size=0.3, random_state=42)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).to(device)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True

    model = GNNAnomalyDetector(data.num_node_features).to(device)
    label_tensor = data.y[data.train_mask]
    class_weights = 1.0 / torch.bincount(label_tensor).float()
    class_weights /= class_weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    losses, accs, f1s, times = [], [], [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            _, pred = out.max(dim=1)
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()
            accs.append(accuracy_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred))

        print(f\"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Acc: {accs[-1]:.4f} | F1: {f1s[-1]:.4f}\")

    return model, pred.cpu().numpy(), y_np, losses, accs, f1s

def evaluate_model(y_true, y_pred):
    print(\"\\n✅ Final Classification Report:\")
    print(classification_report(y_true, y_pred, target_names=[\"Healthy\", \"Compromised\"]))
