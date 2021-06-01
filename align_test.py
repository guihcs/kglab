from pyrdf2vec.graphs import KG
from kge_align import KGEA
from test_utils import getAligns, getCfm, getOntologies
import pandas as pd
import matplotlib.pyplot as plt

def test(k1p, k2p, t):
    kg1 = KG(k1p)
    kg2 = KG(k2p)

    kgea = KGEA(kg1=kg1, kg2=kg2)
    kgea.init()
    kgea.fit(walks=w, epochs=e)
    kgea.alignEmbeddings(threshold=ap, epochs=e)
    aligns = kgea.getAligns(threshold=tp)

    talings = getAligns(t)

    return getCfm(aligns, talings), *kgea.stats(), len(talings), len(aligns)


ap = 0.95
tp = 0.9
w = 2
e = 10

data = pd.DataFrame(columns=['ref', 'kg1', 'kg2', 'correct', 'ref-count', 'aligns'])

for reference, kg1, kg2 in getOntologies():
    res = test(kg1, kg2, reference)
    data = data.append({
        'ref': reference.split('/')[1],
        'kg1': res[1],
        'kg2': res[2],
        'correct': res[0],
        'ref-count': res[3],
        'aligns': res[4],
        'precision': res[0] / res[4],
        'recall': res[0] / res[3],
        'f-measure': (res[0] / res[4] * res[0] / res[3]) / (res[0] / res[4] + res[0] / res[3]),
    }, ignore_index=True)



print(data)
print(data[['precision', 'recall', 'f-measure']].mean())