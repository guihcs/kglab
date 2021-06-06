import math
import re

import numpy as np
from progress.bar import IncrementalBar
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker

# Ensure the determinism of this script by initializing a pseudo-random number.
RANDOM_STATE = 22


def getEntityNames(kg):
    return list(set([n.name for n in kg._entities if re.search(r'http://\w+#\w+', n.name)]))


def getEmbeddings(kg, entities, w=2, e=10):
    transform = RDF2VecTransformer(
        # Ensure random determinism for Word2Vec.
        # Must be used with PYTHONHASHSEED.
        Word2Vec(workers=2, epochs=e),
        # Extract all walks with a maximum depth of 5 for each entity using two
        # processes and use a random state to ensure that the same walks are
        # generated for the entities.
        walkers=[RandomWalker(w, None, n_jobs=2, random_state=RANDOM_STATE)],
        verbose=1,
    )

    return transform.fit_transform(kg, entities=entities)


def getEmbeddingSimilarity(e1, e2):
    vector_1 = np.array(e1)
    vector_2 = np.array(e2)
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = unit_vector_1 @ unit_vector_2
    dot_product = min(max(-1, dot_product), 1)
    return 1 - np.arccos(dot_product) / math.pi


def getEmbeddingMap(entities, embeddings):
    em = {}
    for i in range(len(entities)):
        em[entities[i]] = embeddings[i]

    return em


def getRotationMatrix(A, B):
    N = np.outer(B, A)
    U, S, V = np.linalg.svd(N)
    return U @ V


def alignEmbeddings(aligns, e2, em1, em2, e=3):
    errors = []

    bar = IncrementalBar('Aligning...', max=e)
    for _ in range(e):
        mse = 0
        for align in aligns:
            s = getEmbeddingSimilarity(em1[align[0]], em2[align[1]])
            mse += pow(1 - s, 2)
            r = getRotationMatrix(em1[align[0]], em2[align[1]])
            for e in e2:
                em2[e] = em2[e] @ r

        errors.append(math.sqrt(mse))
        bar.next()

    bar.finish()
    return errors
