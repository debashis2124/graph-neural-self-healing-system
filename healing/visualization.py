import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_curves(losses, accs, f1s, epoch_times, EPOCHS):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), losses, marker='o', color='blue')
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(range(1, EPOCHS + 1), accs, marker='s', color='maroon')
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(range(1, EPOCHS + 1), f1s, marker='^', color='purple')
    plt.title("F1 Score"); plt.xlabel("Epoch"); plt.ylabel("F1 Score")

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), epoch_times, marker='d', color='green')
    plt.title("Epoch Duration"); plt.xlabel("Epoch"); plt.ylabel("Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_healing_effect(before_healthy, before_compromised, after_healthy, after_compromised):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].bar(["Healthy", "Compromised"], [before_healthy, before_compromised], color=['blue', 'orange'])
    ax[0].set_title("Before Healing")
    ax[1].bar(["Healthy", "Compromised"], [after_healthy, after_compromised], color=['green', 'red'])
    ax[1].set_title("After Healing")
    plt.suptitle("Healing Effect on Node Classification")
    plt.tight_layout()
    plt.show()

def plot_classification_bar(TN, FP, TP, FN):
    labels = ['True Healthy Retained', 'Healthy Removed', 'Compromised Removed', 'Compromised Retained']
    values = [TN, FP, TP, FN]
    colors = ['green', 'orange', 'blue', 'red']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 5, f'{int(height)}', ha='center')
    plt.title("Healing Classification Breakdown")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

def plot_healing_confusion(true_labels, removed_mask):
    healing_preds = np.zeros_like(true_labels)
    healing_preds[removed_mask] = 1
    cm = confusion_matrix(true_labels, healing_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Compromised"]).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Healing vs True")
    plt.show()

def plot_graphs(G, healed_G, y_np):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    nx.draw(G, node_color=y_np, with_labels=False, node_size=20, cmap=plt.cm.coolwarm)
    plt.title("Original Graph")
    plt.subplot(1, 2, 2)
    nx.draw(healed_G, with_labels=False, node_size=20, node_color='lightgreen')
    plt.title("Healed Graph")
    plt.tight_layout()
    plt.show()
