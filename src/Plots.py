from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import scipy.cluster.hierarchy as sch

def histograms(data, directory):
    data.hist()
    plt.show
    plt.savefig(directory)

def scatter_plot_matrix(data, directory):
    scatter_matrix(data)
    plt.show()
    plt.savefig(directory)

def scatter_plot(x, y, xlabel, ylabel, title, directory):
    plt.scatter(x, y, alpha = 0.5, s = 70)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.savefig(directory)

def scree_plot(pc_values, explained_variance, xlabel, ylabel, title, directory, color):
    plt.plot(pc_values, explained_variance, 'o-', linewidth = 2, color = color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    plt.savefig(directory)

def plot_of_actual_and_predicted(X, y, predictions, xlabel, ylabel, title, directory):
    plt.scatter(X, y, color = 'blue', label = 'Actual')
    plt.scatter(X, predictions, color = 'red', label = 'Predicted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(directory)

def plot_confusion_matrix(cm, xlabel, ylabel, title, directory):
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.savefig(directory)

def plot_elbow_method(range_clusters, inertia, directory):
    plt.plot(range_clusters, inertia, marker='o')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()
    plt.savefig(directory)

def plot_clusters(pca_data, color, directory):
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c = color)
    plt.title('Clustering of Student Performance')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    plt.savefig(directory)

def plot_dendrogram(data, directory):
    dendrogram = sch.dendrogram(sch.linkage(data, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Students')
    plt.ylabel('Euclidean distances')
    plt.show()
    plt.savefig(directory)
