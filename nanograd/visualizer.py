#!/usr/bin/python3
import graphviz

class Viz:
    def __init__(self, root):
        self.format = 'png'
        self.rankdir = 'LR'
        self.dag = graphviz.Digraph(format=self.format, graph_attr={'rankdir': self.rankdir})
        nodes, edges = self.trace(root)
        for n in nodes:
            self.dag.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.value, n.grad), shape='record')
            if n.operation.value:
                self.dag.node(name=str(id(n)) + n.operation.value, label=n.operation.value)
                self.dag.edge(str(id(n)) + n.operation.value, str(id(n)))
    
        for n1, n2 in edges:
            self.dag.edge(str(id(n1)), str(id(n2)) + n2.operation.value)

    def trace(self, root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v.parents:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges

    def view(self):
        self.dag.view()
