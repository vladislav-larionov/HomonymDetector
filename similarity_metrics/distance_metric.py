from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
# ambiguity_filtered_by_3_samples:
# hamming
# Total: 133/522
# Total used words: 32/32
# canberra
# Total: 227/522
# Total used words: 32/32
# braycurtis
# Total: 221/522
# Total used words: 32/32
# jaccard, matching
# Total: 250/522
# Total used words: 32/32

# homonyms_ru:
# euclidean
# Total: 401/856
# Total used words: 116/116
# manhattan
# Total: 396/856
# Total used words: 116/116
# chebyshev
# Total: 420/856
# Total used words: 116/116
# minkowski
# Total: 401/856
# Total used words: 116/116
# hamming
# Total: 546/856
# Total used words: 116/116
# canberra
# Total: 453/856
# Total used words: 116/116
# braycurtis
# Total: 455/856
# Total used words: 116/116
def compare_by_sklearn(X, Y, metric='braycurtis'):
    dist = DistanceMetric.get_metric(metric)
    # return cosine_similarity(X, Y)
    # return dist.pairwise(X, Y)
    return dist.pairwise(X.reshape(1, -1), Y.reshape(1, -1))

