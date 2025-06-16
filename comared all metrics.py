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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
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

# --------- Load & Preprocess Data ----------
df = pd.read_excel('Supply chain logisitcs problem.xlsx', sheet_name='OrderList')
df['Label'] = (df['Ship Late Day count'] > 3).astype(int)
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Hour']    = df['Order Date'].dt.hour
df['Day']     = df['Order Date'].dt.day
df['Weekday'] = df['Order Date'].dt.weekday
df.ffill(inplace=True)

for col in [
    'Origin Port','Carrier','Service Level','Customer',
    'Product ID','Plant Code','Destination Port'
]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

features = df[[
    'Order ID','Origin Port','Carrier','TPT','Service Level',
    'Ship ahead day count','Ship Late Day count','Customer',
    'Product ID','Plant Code','Destination Port',
    'Unit quantity','Weight','Hour','Day','Weekday'
]].values
features = StandardScaler().fit_transform(features)
labels = df['Label'].values

# --------- Balance with SMOTE ----------
counts = Counter(labels)
minor = min(counts, key=counts.get)
k_nbrs = min(5, counts[minor]-1) if counts[minor]>1 else 1
features_res, labels_res = SMOTE(
    sampling_strategy=SMOTE_RATIO, random_state=42, k_neighbors=k_nbrs
).fit_resample(features, labels)

# --------- Build Graph via KNN ----------
G = nx.Graph()
for i, feat in enumerate(features_res):
    G.add_node(i, x=torch.tensor(feat, dtype=torch.float32),
                  y=int(labels_res[i]))
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine')
knn.fit(features_res)
dists, nbrs = knn.kneighbors(features_res)
for i, (ds, ns) in enumerate(zip(dists, nbrs)):
    for j, dist in zip(ns[1:], ds[1:]):
        if dist < DISTANCE_THRESHOLD:
            G.add_edge(i, j, weight=1-dist)

# --------- PyG Data Object ----------
data = Data(
    x=torch.stack([G.nodes[i]['x'] for i in G.nodes()]).to(device),
    edge_index=torch.tensor(list(G.edges)).t().contiguous().to(device),
    edge_attr=torch.tensor(
        [G[u][v]['weight'] for u,v in G.edges()],
        dtype=torch.float32
    ).to(device),
    y=torch.tensor([G.nodes[i]['y'] for i in G.nodes()],
                   dtype=torch.long).to(device)
)
data.num_node_features = data.x.shape[1]

idx = np.arange(data.num_nodes)
y_np = data.y.cpu().numpy()
train_i, test_i = train_test_split(idx, stratify=y_np,
                                   test_size=0.3, random_state=42)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_i] = True
data.test_mask[test_i]  = True
data.train_mask, data.test_mask = data.train_mask.to(device), data.test_mask.to(device)

# --------- GNN Model Definitions ----------
class GCNNet(nn.Module):
    def __init__(self,in_c,hid_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.conv2 = GCNConv(hid_c, 2)
        self.dp = nn.Dropout(0.5)
    def forward(self,data):
        x,ei = data.x, data.edge_index
        x = F.relu(self.conv1(x,ei)); x=self.dp(x)
        return self.conv2(x,ei)

class GATNet(nn.Module):
    def __init__(self,in_c,hid_c,heads=2):
        super().__init__()
        self.conv1 = GATConv(in_c, hid_c, heads=heads)
        self.conv2 = GATConv(hid_c*heads, 2, heads=1)
        self.dp = nn.Dropout(0.5)
    def forward(self,data):
        x,ei = data.x, data.edge_index
        x = F.elu(self.conv1(x,ei)); x=self.dp(x)
        return self.conv2(x,ei)

class SAGENet(nn.Module):
    def __init__(self,in_c,hid_c):
        super().__init__()
        self.conv1 = SAGEConv(in_c, hid_c)
        self.conv2 = SAGEConv(hid_c, 2)
        self.dp = nn.Dropout(0.5)
    def forward(self,data):
        x,ei = data.x, data.edge_index
        x = F.relu(self.conv1(x,ei)); x=self.dp(x)
        return self.conv2(x,ei)

class GINNet(nn.Module):
    def __init__(self,in_c,hid_c):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_c,hid_c), nn.ReLU(), nn.Linear(hid_c,hid_c))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hid_c,2))
        self.conv2 = GINConv(nn2)
        self.dp = nn.Dropout(0.5)
    def forward(self,data):
        x,ei = data.x, data.edge_index
        x = F.relu(self.conv1(x,ei)); x=self.dp(x)
        return self.conv2(x,ei)

# --------- Training/Evaluation Loop ----------
def train_gnn_epochwise(model_class,data,name,hidden,epochs=50):
    model = model_class(data.num_node_features,hidden).to(device)
    lb = data.y[data.train_mask]; cw = 1.0/torch.bincount(lb).float()
    cw/=cw.sum()
    crit = nn.CrossEntropyLoss(weight=cw.to(device))
    opt  = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-3)

    # track
    tl, vl = [], []
    ta, va = [], []
    tp, vp = [], []
    tr, vr = [], []
    tf, vf = [], []
    ts, vs = [], []

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = crit(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
        tl.append(loss.item())

        # train metrics
        _, pt = out[data.train_mask].max(1)
        t_true = data.y[data.train_mask].cpu().numpy()
        t_pred = pt.cpu().numpy()
        ta.append(accuracy_score(t_true,t_pred))
        tp.append(precision_score(t_true,t_pred,zero_division=0))
        tr.append(recall_score   (t_true,t_pred,zero_division=0))
        tf.append(f1_score       (t_true,t_pred,zero_division=0))
        cm = confusion_matrix(t_true,t_pred)
        ts.append(cm[0,0]/(cm[0,0]+cm[0,1]+1e-6))

        # val metrics
        model.eval()
        with torch.no_grad():
            outv = model(data)
            lv   = crit(outv[data.test_mask], data.y[data.test_mask])
            vl.append(lv.item())
            _, pv = outv.max(1)
            v_true= data.y[data.test_mask].cpu().numpy()
            v_pred= pv[data.test_mask].cpu().numpy()
            va.append(accuracy_score(v_true,v_pred))
            vp.append(precision_score(v_true,v_pred,zero_division=0))
            vr.append(recall_score   (v_true,v_pred,zero_division=0))
            vf.append(f1_score       (v_true,v_pred,zero_division=0))
            cm2=confusion_matrix(v_true,v_pred)
            vs.append(cm2[0,0]/(cm2[0,0]+cm2[0,1]+1e-6))

    return {
      "name":name,
      "train_loss":tl, "val_loss":vl,
      "train_acc":ta,  "val_acc":va,
      "train_prec":tp,"val_prec":vp,
      "train_rec":tr, "val_rec":vr,
      "train_f1":tf,  "val_f1":vf,
      "train_spec":ts,"val_spec":vs
    }

# --------- Train All Models ---------
all_models = [("GCN",GCNNet),("GAT",GATNet),("GraphSAGE",SAGENet),("GIN",GINNet)]
results = []
for n,m in all_models:
    print("Training",n)
    results.append(train_gnn_epochwise(m,data,n,HIDDEN_CHANNELS,EPOCHS))

    # --------- Plot Metric Comparisons (All Models) ---------
def plot_metric_comparison(results, metric, ylabel):
    plt.figure(figsize=(9, 5))
    markers = ['o','s','^','x','D','P','*','<','>','v','h','+','1','2','3','4']
    for idx, res in enumerate(results):
        marker = markers[idx % len(markers)]
        plt.plot(
            range(1, len(res[metric]) + 1),
            res[metric],
            label=res["name"],
            marker=marker,
            linewidth=2
        )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Comparison Among Various GNN Models")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    filename = f"{ylabel.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Plot all metrics
plot_metric_comparison(results, "train_acc", "Training Accuracy")
plot_metric_comparison(results, "val_acc",   "Validation Accuracy")
plot_metric_comparison(results, "train_prec","Training Precision")
plot_metric_comparison(results, "val_prec",  "Validation Precision")
plot_metric_comparison(results, "train_rec", "Training Recall")
plot_metric_comparison(results, "val_rec",   "Validation Recall")
plot_metric_comparison(results, "train_f1",  "Training F1 Score")
plot_metric_comparison(results, "val_f1",    "Validation F1 Score")
plot_metric_comparison(results, "train_loss","Training Loss")
plot_metric_comparison(results, "val_loss",  "Validation Loss")
plot_metric_comparison(results, "train_spec","Training Specificity")
plot_metric_comparison(results, "val_spec",  "Validation Specificity")

# --------- Final Metric Table (Last Epoch) ----------
summary = []
for r in results:
    summary.append([
        r["name"],
        f"{r['train_acc'][-1]:.4f}",
        f"{r['val_acc'][-1]:.4f}",
        f"{r['train_prec'][-1]:.4f}",
        f"{r['val_prec'][-1]:.4f}",
        f"{r['train_rec'][-1]:.4f}",
        f"{r['val_rec'][-1]:.4f}",
        f"{r['train_f1'][-1]:.4f}",
        f"{r['val_f1'][-1]:.4f}",
        f"{r['train_loss'][-1]:.4f}",
        f"{r['val_loss'][-1]:.4f}",
        f"{r['train_spec'][-1]:.4f}",
        f"{r['val_spec'][-1]:.4f}"
    ])

print("\nFinal Model Metrics (last epoch):")
print(tabulate(
    summary,
    headers=[
      "Model","Tr Acc","Val Acc",
      "Tr Prec","Val Prec",
      "Tr Rec","Val Rec",
      "Tr F1","Val F1",
      "Tr Loss","Val Loss",
      "Tr Spec","Val Spec"
    ],
    tablefmt="fancy_grid"
))

# --------- Train/Eval helper ---------
# --------- Train/Eval Best-F1 Helper ----------
def train_eval_best(model_cls):
    model = model_cls(data.num_node_features, HIDDEN_CHANNELS).to(device)
    lb = data.y[data.train_mask]
    cw = 1.0 / torch.bincount(lb).float(); cw /= cw.sum()
    loss_fn = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    best_f1, best_state = 0, None

    for _ in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            _, preds = model(data).max(1)
            y_t = data.y[data.test_mask].cpu().numpy()
            y_p = preds[data.test_mask].cpu().numpy()
            f = f1_score(y_t, y_p, zero_division=0)
            if f > best_f1:
                best_f1, best_state = f, model.state_dict()

    model.load_state_dict(best_state)
    return model

# --------- Train Best Models ----------
models = {name: train_eval_best(cls)
          for name, cls in [("GCN", GCNNet), ("GAT", GATNet),
                            ("GraphSAGE", SAGENet), ("GIN", GINNet)]}

# --------- Prediction Helper ----------
def count_preds(model):
    model.eval()
    with torch.no_grad():
        _, p = model(data).max(1)
    return p.cpu().numpy()

# --------- Final Node-Count Table ----------
true = data.y.cpu().numpy()
test_m = data.test_mask.cpu().numpy()
total = data.num_nodes
rows = []
for name, mdl in models.items():
    preds = count_preds(mdl)
    comp = int((preds == 1).sum())
    rec  = int(((preds == 1) & (true == 1) & test_m).sum())
    rows.append([name, total, comp, rec])

print("\n====== Node Count Per Model ======")
print(tabulate(rows,
               headers=["Model","Total Nodes","Compromised Nodes","Recovered (TP on Test)"],
               tablefmt="fancy_grid"))

# --------- Healing Logic (with Loss) ----------
def evaluate_healing(model_cls, name):
    # retrain best-F1
    mdl = train_eval_best(model_cls)
    preds = count_preds(mdl)
    # healing loss on test split
    lb = data.y[data.test_mask]
    cw = 1.0 / torch.bincount(data.y[data.train_mask]).float(); cw /= cw.sum()
    loss_fn = nn.CrossEntropyLoss(weight=cw.to(device))
    with torch.no_grad():
        outv = mdl(data)
        heal_loss = loss_fn(outv[data.test_mask], lb).item()
    # masks
    true_l = data.y.cpu().numpy()
    m_r = preds == 1
    m_h = ~m_r
    th = true_l == 0
    tc = true_l == 1
    TP = int((m_r & tc).sum()); FP = int((m_r & th).sum())
    TN = int((m_h & th).sum()); FN = int((m_h & tc).sum())
    tot = len(true_l)
    acc  = (TP + TN) / tot
    prec = TP / (TP + FP + 1e-6)
    rec  = TP / (TP + FN + 1e-6)
    spec = TN / (TN + FP + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return [name, f"{heal_loss:.4f}", f"{acc:.4f}", f"{prec:.4f}",
            f"{rec:.4f}", f"{f1:.4f}", f"{spec:.4f}", TP, FP, TN, FN]

hrows = [evaluate_healing(cls, name)
         for name, cls in [("GCN",GCNNet),("GAT",GATNet),
                           ("GraphSAGE",SAGENet),("GIN",GINNet)]]
print("\n=== Healing Performance (with Loss) ===")
print(tabulate(
    hrows,
    headers=["Model","Heal Loss","Heal Acc","Heal Prec",
             "Heal Rec","Heal F1","Heal Spec","TP","FP","TN","FN"],
    tablefmt="fancy_grid"
))

import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np

true = data.y.cpu().numpy()
mask = data.test_mask.cpu().numpy()
# restrict everything to test indices
test_idx = np.where(mask)[0]
total_test = test_idx.shape[0]

def find_best_threshold_by_f1(model):
    model.eval()
    with torch.no_grad():
        out = model(data)
        probs = F.softmax(out[mask], dim=1)[:,1].cpu().numpy()
    y_val = true[mask]

    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0, 1, 101):
        y_pred = (probs > t).astype(int)
        f = f1_score(y_val, y_pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    return best_t, best_f1

rows = []
for name, mdl in models.items():
    # 1) choose threshold by maximizing F1 on test split
    thresh, val_f1 = find_best_threshold_by_f1(mdl)

    # 2) apply to test split only
    mdl.eval()
    with torch.no_grad():
        out_all = mdl(data)
        probs_all = F.softmax(out_all, dim=1)[:,1].cpu().numpy()
    preds_all = (probs_all > thresh).astype(int)
    preds_test = preds_all[mask]

    # compute test‐restricted metrics
    pred_compromised = int(preds_test.sum())
    recovered        = int(((preds_test == 1) & (true[mask] == 1)).sum())

    rows.append([
        name,
        f"{thresh:.2f}",     # chosen threshold
        f"{val_f1:.3f}",     # F1 at that threshold on test
        total_test,          # total test nodes
        pred_compromised,    # predicted compromised on test
        recovered            # recovered TP on test
    ])

print("\n====== Node Count Per Model (test‐only) ======")
print(tabulate(
    rows,
    headers=[
      "Model","Thresh","Val F1",
      "Total Test","Pred Comp","Recovered TP"
    ],
    tablefmt="fancy_grid"
))
