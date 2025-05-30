import numpy as np
from tabulate import tabulate

def apply_graph_healing(G, data, all_pred):
    healed_G = G.copy()
    for i, pred_label in enumerate(all_pred):
        if pred_label == 1 and i in healed_G:
            healed_G.remove_node(i)
    return healed_G

def evaluate_healing_effect(data, healed_G, all_pred):
    true_labels = data.y.cpu().numpy()
    total_nodes = len(true_labels)
    kept_mask = np.zeros(total_nodes, dtype=bool)
    healed_nodes = list(healed_G.nodes())
    kept_mask[healed_nodes] = True
    removed_mask = ~kept_mask

    TP = np.sum(removed_mask & (true_labels == 1))
    FP = np.sum(removed_mask & (true_labels == 0))
    TN = np.sum(kept_mask & (true_labels == 0))
    FN = np.sum(kept_mask & (true_labels == 1))

    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    audit_table = [
        ["True Healthy Retained ✅", TN, f"{100 * TN / total:.2f}%"],
        ["Healthy Removed ❌", FP, f"{100 * FP / total:.2f}%"],
        ["Compromised Removed ✅", TP, f"{100 * TP / total:.2f}%"],
        ["Compromised Retained ❌", FN, f"{100 * FN / total:.2f}%"]
    ]

    metrics_table = [
        ["Accuracy", accuracy],
        ["Precision", precision],
        ["Recall", recall],
        ["Specificity", specificity],
        ["F1-Score", f1]
    ]

    print("\n📝 Healing Metrics Report:")
    print(tabulate(audit_table, headers=["Metric", "Count", "Percentage"], tablefmt="fancy_grid"))
    print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))
