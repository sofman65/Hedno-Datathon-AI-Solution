import numpy as np
from hdbscan import HDBSCAN
from hdbscan import approximate_predict

class GeoClustering:
    def __init__(self, data):
        self.data = data
        self.hdbscan = None

    def deg2rad(self, degrees):
        return np.radians(degrees)

    def cluster(self, min_samples=2, chunk_size=50000):
        # Convert coordinates to radians
        rad_coords = self.deg2rad(self.data)

        # Divide the dataset into smaller chunks
        num_chunks = int(np.ceil(rad_coords.shape[0] / chunk_size))
        chunked_data = np.array_split(rad_coords, num_chunks)

        # Initialize a new column for cluster labels
        self.data['cluster_labels'] = -1

        # Instantiate HDBSCAN using the haversine metric
        self.hdbscan = HDBSCAN(min_samples=min_samples, metric='haversine', core_dist_n_jobs=-1, prediction_data=True)
        
        # Process each chunk separately
        for idx, chunk in enumerate(chunked_data):

            # Fit HDBSCAN to the chunk
            clusters = self.hdbscan.fit(chunk)

            # Store the cluster labels for this chunk
            chunk_indices = chunk.index
            self.data.loc[chunk_indices, 'cluster_labels'] = clusters.labels_

        return self.data

    def predict(self, new_points):
        if self.hdbscan is None:
            raise ValueError("You must call .cluster() before calling .predict()")

        # Convert new_points to radians
        rad_new_points = self.deg2rad(new_points)

        # Predict the cluster labels for the new points using approximate_predict()
        labels, strengths = approximate_predict(self.hdbscan, rad_new_points)
        
        return labels
        
    def save(self, file_path):
        if self.hdbscan is None:
            raise ValueError("You must call .cluster() before calling .save()")
        
        with open(file_path, 'wb') as f:
            pickle.dump(self.hdbscan, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.hdbscan = pickle.load(f)