from sklearn.cluster import KMeans

def get_kmeans_labels(X, k):
	kmeans = KMeans(n_clusters=k, random_state=111)
	kmeans.fit(X)
	return kmeans.labels_