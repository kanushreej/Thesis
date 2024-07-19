# Required imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import umap

# Load the dataset
file_path = '/Users/kanushreejaiswal/Desktop/Thesis/Analyses/User Data/usersUS_1_preprocessed.csv'
df = pd.read_csv(file_path)

# Selecting the relevant features for clustering
features = [
    'pro_immigration', 'anti_immigration', 'pro_climateAction', 'anti_climateAction',
    'public_healthcare', 'private_healthcare', 'pro_israel', 'pro_palestine',
    'pro_middle_low_tax', 'pro_wealthy_corpo_tax'
]

# Extracting the features from the dataframe
X = df[features]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Performing UMAP to reduce to 2 components for visualization
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)

# Performing Hierarchical Clustering
linked = linkage(X_scaled, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
