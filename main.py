import xml.etree.ElementTree as ET
import random

class Node:

    def __init__(self):
        self.tag = None
        self.attributes = dict()
        self.text = None
        self.child = []

    def key(self):
        return list(self.attributes.values())[0] if not self.isMeta() else self.tag

    def isMeta(self):
        return len(self.attributes) <= 0


class KG:

    def __init__(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        self.entities = dict()
        nodes = self._getNodes(root)
        self._linkGraph(nodes)

    def _getNodes(self, child):
        nodes = []
        for c in child:
            n = Node()
            n.tag = c.tag.split('}')[1]
            for a in c.attrib.items():
                n.attributes[a[0].split('}')[1]] = a[1].split('#')[-1]
            n.text = c.text
            n.child = self._getNodes(c)
            nodes.append(n)
        return nodes

    def _linkGraph(self, nodes):
        lg = []
        for n in nodes:
            if n.key() not in self.entities:
                self.entities[n.key()] = n

            ln = self.entities[n.key()]
            ln.child = self._linkGraph(n.child)
            lg.append(ln)
        return lg

    def getEntities(self):
        en = []
        for n in self.entities.values():
            if n.tag in ['Class', 'ObjectProperty', 'DatatypeProperty', 'FunctionalProperty',
                         'InverseFunctionalProperty']:
                en.append(n.key())

        return en

    def union(self, kg, identities):
        for i in identities:
            self.entities[i].child += kg.entities[i].child

        for n in kg.entities.values():
            if n.key() not in self.entities:
                self.entities[n.key()] = n

        ups = set()

        self._updateNodes(self.entities.values(), ups)

    def _updateNodes(self, nodes, ups):
        ln = []
        for n in nodes:
            if not n.isMeta() and n in ups:
                continue
            l = self.entities[n.key()]
            ups.add(l)
            l.child = self._updateNodes(l.child, ups)
            ln.append(l)

        return ln


def getNodeAligns(kg1, kg2):
    aligns = set()

    for n1 in kg1.getEntities():
        for n2 in kg2.getEntities():
            if n1.lower() != n2.lower() or kg1.entities[n1].isMeta() or kg2.entities[n2].isMeta():
                continue
            aligns.add(n1)

    return list(aligns)


def deepWalk(kg, maxPath=2, maxRetry=1):
    walks = []
    for n in kg.entities.values():
        for i in range(maxRetry):

            node = n
            walk = []
            for c in range(maxPath):
                walk.append(node.key())
                if len(node.child) < 1:
                    break
                node = random.choice(node.child)
            walks.append(walk)

    return walks

cmtKG = KG('conference/cmt.owl')
conferenceKG = KG('conference/conference.owl')
print(len(cmtKG.getEntities()))
print(len(conferenceKG.getEntities()))
aligns = getNodeAligns(cmtKG, conferenceKG)

cmtKG.union(conferenceKG, aligns)
print(len(cmtKG.getEntities()))
walks = deepWalk(cmtKG, 5, 5)
for w in walks:
    print(w)

print(len(walks))