from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from src.Plots import plot_elbow_method, plot_clusters, plot_dendrogram
import matplotlib.pyplot as plt

class Models:
    def __init__(self, data, original_data):
        self.data = data
        self.original_data = original_data

    def KMeansClustering(self):
        # Calculate the sum of squared distances for a range of cluster numbers
        inertia = []
        range_clusters = range(1, 11)
        for k in range_clusters:
            kmeans = KMeans(n_clusters = k, random_state = 42)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)

        # Print the inertia values for each cluster count
        print("Inertia values for different number of clusters:")
        for k, val in zip(range_clusters, inertia):
            print(f"Clusters: {k}, Inertia: {val}")

        plot_elbow_method(range_clusters, inertia, f'./Plots/Elbow_method_results.png')

        optimal_clusters = 4 # Assuming the optimal number of clusters is determined from the Elbow Method (e.g., 4 clusters)
        
        kmeans = KMeans(n_clusters = optimal_clusters, random_state = 42) # Apply K-means clustering with the optimal number of clusters
        kmeans.fit(self.data)
        self.original_data['Cluster_kmeans'] = kmeans.labels_

        # Analyze the clusters
        cluster_summary = self.original_data.groupby('Cluster_kmeans').mean()
        print("\nCluster Summary:")
        print(cluster_summary)

        pca = PCA(n_components = 2)
        pca_data = pca.fit_transform(self.data)
        plot_clusters(pca_data, kmeans.labels_, f'./Plots/Kmeans_clusters.png')

    def HierarchicalClustering(self):
        plot_dendrogram(self.data, f'./Plots/Dendrogram.png')

        # Fit Hierarchical Clustering model
        hc = AgglomerativeClustering(n_clusters = 4, metric = 'euclidean', linkage = 'ward')
        hc.fit(self.data)
        self.original_data['Cluster_hc'] = hc.labels_

        # Analyze the clusters
        cluster_summary = self.original_data.groupby('Cluster_hc').mean()
        print("\nCluster Summary:")
        print(cluster_summary)

        pca = PCA(n_components = 2)
        pca_data = pca.fit_transform(self.data)
        plot_clusters(pca_data, hc.labels_, f'./Plots/HC_clusters.png')

    def DBScanClustering(self):
        # Experiment with different eps and min_samples values
        eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        min_samples_values = [3, 5, 7, 10]

        pca = PCA(n_components = 2)
        pca_data = pca.fit_transform(self.data)

        # Create a plot to visualize all DBSCAN cases
        fig, axes = plt.subplots(len(eps_values), len(min_samples_values), figsize = (20, 15))
        fig.suptitle('DBSCAN Clustering with Different Parameters', fontsize = 16)

        for i, eps in enumerate(eps_values):
            for j, min_samples in enumerate(min_samples_values):
                dbscan = DBSCAN(eps = eps, min_samples = min_samples)
                dbscan_labels = dbscan.fit_predict(self.data)
        
                self.original_data['Cluster_dbscan'] = dbscan.labels_ # Add the cluster labels to the original dataframe
                num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0) # Count the number of clusters (excluding noise)
        
                print(f"DBSCAN with eps = {eps} and min_samples = {min_samples}")
                print(f"Number of clusters: {num_clusters}")
                print(f"Cluster labels: {set(dbscan.labels_)}\n")

                # Analyze the clusters
                cluster_summary = self.original_data.groupby('Cluster_dbscan').mean()
                print("\nCluster Summary:")
                print(cluster_summary)

                # Plot the results
                ax = axes[i, j]
                scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c = dbscan_labels, cmap = 'viridis', marker = 'o')
                ax.set_title(f'eps = {eps}, min_samples = {min_samples}')
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')

        plt.tight_layout(rect = [0, 0, 1, 0.96])
        plt.savefig(f'./Plots/DBSCAN Clustering with Different Parameters.png')
        plt.show()
