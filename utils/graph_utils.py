import networkx as nx
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def build_graph(features, labels, k_neighbors=10, sim_thresh=0.8):
    G = nx.Graph()
    for i in range(len(features)):
        G.add_node(i, x=torch.tensor(features[i], dtype=torch.float32), y=int(labels[i]))

    knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    for i, (dists, neighbors) in enumerate(zip(distances, indices)):
        for j, dist in zip(neighbors[1:], dists[1:]):
            if dist < (1 - sim_thresh):
                G.add_edge(i, j, weight=1 - dist)
    return G

def convert_to_pyg_data(G):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Data()
    data.edge_index = torch.tensor(list(G.edges)).t().contiguous().to(device)
    data.edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges()], dtype=torch.float32).to(device)
    data.x = torch.stack([G.nodes[i]['x'] for i in G.nodes()]).to(device)
    data.y = torch.tensor([G.nodes[i]['y'] for i in G.nodes()], dtype=torch.long).to(device)
    data.num_node_features = data.x.shape[1]
    return data
