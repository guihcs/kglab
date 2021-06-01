import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')


def parseEntities(e1):
    parsedEntities = []

    for e in e1:
        parsedEntities.append(removeStopwords(splitEntity(getEntityNuclei(e))))

    return parsedEntities


def getLexicalAligns(e1, e2, t):
    aligns = []

    parsed_e1_entities = parseEntities(e1)
    parsed_e2_entities = parseEntities(e2)

    for i in range(len(e1)):
        for j in range(len(e2)):
            if len(parsed_e1_entities[i]) <= 0 or len(parsed_e2_entities[j]) <= 0:
                continue
            if stemSimilarity(parsed_e1_entities[i], parsed_e2_entities[j]) > t:
                aligns.append([e1[i], e2[j]])

    return aligns


def getEntityNuclei(e):
    return e.split('#')[1]


def splitEntity(e):
    split = []
    sp = ''
    for i in range(len(e)):
        if e[i].islower() and i + 1 < len(e) and e[i + 1].isupper():
            sp += e[i]
            split += [sp]
            sp = ''
            continue

        if e[i] == '_':
            split += [sp]
            sp = ''
            continue
        sp += e[i]
    split += [sp]
    return split


def removeStopwords(e):
    ps = PorterStemmer()
    return [ps.stem(x.lower()) for x in e if x.lower() not in stopwords.words()]


def stemSimilarity(e1, e2):
    s1 = set(e1)
    s2 = set(e2)

    return 2 * len(s1.intersection(s2)) / (len(s1) + len(s2))
