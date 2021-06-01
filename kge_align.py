from kg_word import *
from kge import *


def embeddingSimilarityError(align, e1, e2):
    mse = 0

    for a in align:
        s = getEmbeddingSimilarity(e1[a[0]], e2[a[1]])
        mse += pow(1 - s, 2)

    return mse / max(len(align), 1)


class KGD:

    def __init__(self, kg):
        self.kg = kg
        self.entities = None
        self.embeddings = None
        self.embeddingsMap = None

    def init(self):
        self.entities = getEntityNames(self.kg)

    def fit(self, walks, epochs):
        self.embeddings = getEmbeddings(self.kg, self.entities, w=walks, e=epochs)[0]
        self.embeddingsMap = getEmbeddingMap(self.entities, self.embeddings)


class KGEA:

    def __init__(self, kg1, kg2):
        self.kgd1 = KGD(kg1)
        self.kgd2 = KGD(kg2)

    def init(self):
        self.kgd1.init()
        self.kgd2.init()

    def fit(self, walks, epochs):
        self.kgd1.fit(walks=walks, epochs=epochs)
        self.kgd2.fit(walks=walks, epochs=epochs)

    def alignEmbeddings(self, threshold, epochs):
        ea = getLexicalAligns(self.kgd1.entities, self.kgd2.entities, threshold)
        preAlignError = embeddingSimilarityError(ea, self.kgd1.embeddingsMap, self.kgd2.embeddingsMap)
        alignErrors = alignEmbeddings(ea,
                                      self.kgd2.entities,
                                      self.kgd1.embeddingsMap,
                                      self.kgd2.embeddingsMap, e=epochs)
        postAlignError = embeddingSimilarityError(ea, self.kgd1.embeddingsMap, self.kgd2.embeddingsMap)

        return preAlignError, alignErrors, postAlignError

    def getAligns(self, threshold):
        pe1 = parseEntities(self.kgd1.entities)
        pe2 = parseEntities(self.kgd2.entities)

        aligns = []
        als = set()

        for i in range(len(self.kgd1.entities)):
            for j in range(len(self.kgd2.entities)):
                e1 = self.kgd1.entities[i]
                e2 = self.kgd2.entities[j]
                if len(pe1[i]) <= 0 or len(pe2[j]) <= 0:
                    continue

                wordSimilarity = stemSimilarity(pe1[i], pe2[j])
                embeddingSimilarity = getEmbeddingSimilarity(self.kgd1.embeddingsMap[e1], self.kgd2.embeddingsMap[e2])

                if max(wordSimilarity, embeddingSimilarity) > threshold and e1 + e2 not in als:
                    aligns.append([e1, e2])
                    als.add(e1 + e2)

        return aligns

    def stats(self):
        return len(self.kgd1.entities), len(self.kgd2.entities)