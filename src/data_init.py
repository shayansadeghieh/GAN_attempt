import networkx as nx
import pandas as pd 
import matplotlib.pyplot as plt

class GraphGenerator:
    '''
    class to generate different graphs 
    '''
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
    
    def generate_graph(self):
        graph = nx.cycle_graph(self.num_nodes)
        return graph

if __name__ == "__main__":
    num_nodes = 10
    graph_init = GraphGenerator(num_nodes)
    graph = graph_init.generate_graph()
    nx.draw(graph, with_labels = True)
    plt.show()

        
    
