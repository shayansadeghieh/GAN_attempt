import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


class DataVisualize:
    '''
    Class to visualize the data for sanity and curiosity 
    '''
    def __init__(self, filepath):
        '''
        Init the following args:
        -filepath
        '''
        #Init files 
        self.filepath = filepath
        self.dataset = pd.read_csv(self.filepath)

    def graph_properties(self):
        graph = nx.from_pandas_edgelist(self.dataset, source = 'node_1', target = 'node_2')
        return graph
        
    # def __call__(self):
    #     nx.draw(self.graph_properties())
    #     plt.show()

if __name__ == "__main__":
    filepath = "/Users/shayansadeghieh/Downloads/facebook_clean_data/company_edges.csv"
    data = DataVisualize(filepath)
    graph = data.graph_properties()
    
    
    
