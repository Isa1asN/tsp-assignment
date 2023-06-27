class Node:
    def __init__(self, name):
        self.name = name
        self.edges = []
        
    def add_edge(self, node, weight):
        self.edges.append((node, weight))

    def remove_edge(self, node):
        for edge in self.edges:
            if edge[0] == node:
                self.edges.remove(edge)
                break

class Graph:
    def __init__(self):
        self.vertices = {}
        
    def add_node(self, node):
        self.vertices[node.name] = node
        
    def add_edge(self, node1, node2, weight):
        node1.add_edge(node2, weight)
        node2.add_edge(node1, weight)
        
    def remove_node(self, node):
        del self.vertices[node.name]
        for vertex in self.vertices.values():
            vertex.remove_edge(node)
    
    def remove_edge(self, node1, node2):
        node1.remove_edge(node2)
        node2.remove_edge(node1)
        
    def search(self, item):
        for node in self.vertices.values():
            if node.name == item:
                return True
        return False

def load_graph_from_file(filename):
    graph = Graph()
    with open(filename, 'r') as f:
        for line in f:
            node1, node2, weight = line.strip().split()
            if node1 not in graph.vertices:
                graph.add_node(Node(node1))
            if node2 not in graph.vertices:
                graph.add_node(Node(node2))
            graph.add_edge(graph.vertices[node1], graph.vertices[node2], int(weight))
    return graph

def path_cost_calculator(graph, path):
        total_cost = 0
        for i in range(len(path)-1):
            node1 = graph.vertices[path[i]]
            node2 = graph.vertices[path[i+1]]
            edge_weight = 0
            for edge in node1.edges:
                if edge[0] == node2:
                    edge_weight = edge[1]
                    break
            total_cost += edge_weight
        return total_cost