import numpy as np
from gensim import matutils
from numpy import dot
from numpy.linalg import norm


def similarity_cosine_w2v(vec1, vec2):
    cosine_similarity = np.dot(matutils.unitvec(vec1), matutils.unitvec(vec2))
    return cosine_similarity

def similarity_cosine_numpy(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

