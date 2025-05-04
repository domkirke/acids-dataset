import torch, numpy as np
from sklearn.cluster import MiniBatchKMeans

def hash_from_clustering(dataset, feature, n_centroids, n_examples=None, write=False, **kwargs):
    assert feature in dataset.features, "feature %s not found in feature_hash"%(feature)
    feature_hash = feature_hash[feature]
    # apply clustering
    if n_examples is None: n_examples = len(dataset)
    features = []
    for i in range(n_examples):
        feature = dataset.get(i, output_pattern=feature)
        if isinstance(feature, torch.Tensor): feature = feature.numpy()
        features.append(feature)

    features = np.stack(features, axis=0)
    kmeans = MiniBatchKMeans(n_clusters=n_centroids, random_state=0, **kwargs)
    kmeans.fit(features)

    feature_hash = {i: [] for i in range(n_centroids)}
    for i in range(len(dataset)):
        feature = dataset.get(i, output_pattern=feature)
        k = kmeans.predict(feature)
        feature_hash[k] = dataset.keys[i]

    if write: 
        dataset.loader.writer.add_feature_hash(feature, feature_hash)
        dataset.loader.writer.append_to_feature_metadata(feature, {'hash_from_cluster': kmeans})
    
    return kmeans
        

