import re
import xml.etree.ElementTree as ET
from os import walk


def getCfm(t1, t2):
    correct = 0
    for c1 in t2:
        for a1 in t1:
            if len(a1) < 2 or len(c1) < 2:
                continue
            if a1[0] == c1[0] and a1[1] == c1[1]:
                correct += 1
                break

    return correct


def getAligns(kgPath):
    tree = ET.parse(kgPath)
    root = tree.getroot()
    aligns = []
    for c in root[0]:
        if c.tag.endswith('map'):
            alc = []
            for cm in c[0]:
                if re.search(r'entity\d$', cm.tag):
                    alc.append(cm.attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'])
            aligns.append(alc)

    return aligns


def getOntologies():
    for path, dirs, files in walk('reference-alignment'):
        for f in files:
            fn = f.split('.')[0].split('-')
            yield f'reference-alignment/{f}', f'conference/{fn[0]}.owl', f'conference/{fn[1]}.owl'
