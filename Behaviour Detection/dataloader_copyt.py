# DATALOADER CLASS to handle the data loading and preprocessing
# We load the .h5 files with the trajectories of DeepLabCut and preprocess them to build the graps

import h5py
import numpy as np
import os
import torch
#import utils as ut
import pandas as pd

from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix


# Class to handle the data for loading and further processing
class DataDLC:
    ''' Class to handle the data for loading and further processing. '''
    def __init__(self, file):
        ''' Constructor of the DataDLC class. It loads the data from the .h5 files and preprocesses it to build the graphs.

            Args:
                file (str): The file to load.'''
        self.file = file
        self.load_data()

    def load_data(self):
        ''' Function that loads the data from the .h5 files and preprocesses it to build the graphs. '''
        
        loaded_tab = pd.read_hdf(self.file) # Load the .h5 file
        # Get the scorers
        scorer = loaded_tab.columns.levels[0]
        if len(scorer) > 1:
            print('More than one scorer in the .h5 file, the scorers are: ', scorer.values)

        #Drop scorer (first level of the columns)
        loaded_tab.columns = loaded_tab.columns.droplevel(0)

        # Save loaded tab
        self.coords = loaded_tab

        self.individuals = loaded_tab.columns.levels[0] # Get the individuals
        self.n_individuals = len(self.individuals) # Get the number of individuals
        self.body_parts = loaded_tab.columns.levels[1] # Get the body parts
        self.n_body_parts = len(self.body_parts) # Get the number of body parts
        self.n_frames = len(loaded_tab) # Get the number of frames
     
                    


class DLCDataLoader(DataLoader):
    ''' The DataLoader class for the DeepLabCut data. It loads the data from the .h5 files and preprocesses it to build the graphs. '''
    
    def __init__(self, root, batch_size, num_workers, device, window_size=None, stride=None):
        ''' Constructor of the DataLoader class. It loads the data from the .h5 files and preprocesses it to build the graphs.

            Args:
                root (str): The root directory of the .h5 files.
                batch_size (int): The batch size.
                num_workers (int): The number of workers for the DataLoader.
                device (torch.device): The device to load the data.
                window_size (int): The window size for the temporal graph.
                stride (int): The stride for the temporal graph.'''

        self.root = root
        self.batch_size = batch_size # Batch size
        self.num_workers = num_workers # Number of workers for the DataLoader
        self.device = device # Device to load the data
        self.window_size = window_size
        self.stride = stride # Stride for the temporal graph

        self.files = [f for f in os.listdir(root) if f.endswith('.h5')]
        self.files.sort()
        self.n_files = len(self.files) # Number of files, i.e. number of spatio-temporal graphs
        self.dataset = []
        self.behaviour = []

        self.load_data()

        print(f"Number of files: {self.n_files}")

        super(DLCDataLoader, self).__init__(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def build_graph(self, data):
        ''' Function that builds the graph from the coordinates of the individuals.

            We have edges between all the nodes of the same individual in the same frame, 
            the nose of the individuals in the same frame and the tail of the individuals in the same frame.
            Also we have edges between the nodes of the same body part accross adjecent frames.

            Args:
                data (DataDLC): The data from the .h5 file.

            Returns:
                node_features (torch.Tensor): The node features of the graph.
                edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].'''
   
        # Get the number of nodes
        n_nodes = data.n_individuals * data.n_body_parts * data.n_frames
        # node-level frame mask grey encoding
        frame_mask = torch.zeros(n_nodes, dtype=torch.int64)
        # Get the number of edges
        # Edges between the nodes of the same individual in the same frame + edges between same body parts in adjecent frames 
        n_edges = data.n_individuals * data.n_body_parts**2 * data.n_frames + data.n_individuals * data.n_body_parts * (data.n_frames - 1)
        # Initialize the node features
        node_features = torch.zeros(n_nodes, 3)
        # Initialize the edge index
        edge_index = torch.zeros(2, n_edges, dtype=int)

        edge = 0
        # Fill the node features
        for i, ind in enumerate(data.individuals):
            for j, bp in enumerate(data.body_parts):
                for k, f in enumerate(range(data.n_frames)):
                    node = i * data.n_body_parts * data.n_frames + j * data.n_frames + k
                    node_features[node, :] = data.coords.loc[f, (ind, bp)].values
                    frame_mask[node] = k

                    # Fill the edge index
                    for l in range(data.n_body_parts):
                        edge_index[0, edge] = node
                        edge_index[1, edge] = i * data.n_body_parts * data.n_frames + l * data.n_frames + k
                        edge += 1

                    if k < data.n_frames - 1:
                        edge_index[0, edge] = node
                        edge_index[1, edge] = node + 1
                        edge += 1

        return node_features, edge_index, frame_mask
        

    def load_data(self):
        i = 0
        for i, file in enumerate(self.files):
            
            print(f"Loading file {file}")
            dlc_data = DataDLC(os.path.join(self.root, file))
            loaded_tab = pd.read_hdf(os.path.join(self.root, file)) # Load the .h5 file
            loaded_tab = loaded_tab.T.reset_index(drop=False).T # Transpose the table
            loaded_tab.columns = loaded_tab.loc["scorer", :] # Set the columns names
            loaded_tab = loaded_tab.iloc[1:] # Remove the first row
            
            # To numpy
            loaded_tab = loaded_tab.to_numpy()
            
            # Check the number of individuals
            individuals = np.unique(loaded_tab[0]) # Get the unique individuals
            n_individuals = len(individuals) # Get the number of individuals
            
            # Check the number of frames
            n_frames = len(loaded_tab) - 3 # 3 first rows are not frames, but metadata
            
            # Build the graph
            # Save the first row, identify the individuals and remove the metadata
            identities = loaded_tab[0]
            loaded_tab = loaded_tab[3:]

            # Convert from object array to float
            loaded_tab = loaded_tab.astype(float)
            coords_indv = torch.zeros(n_individuals, n_frames, loaded_tab.shape[1]//n_individuals) # Initialize the coordinates of the individuals
            for indv in range(n_individuals):
                a = loaded_tab[:,np.where(identities == individuals[indv])]
                coords_indv[indv] = torch.from_numpy(np.squeeze(a, axis=1)) # Get the coordinates of each individual

            if self.window_size is None:
                # Build the graph
                node_features, edge_index, frame_mask = self.build_graph(coords_indv)
                # Build the data object
                data = Data(x=node_features, edge_index=edge_index, y=file, frame_mask=frame_mask)
                self.dataset.append(data)
                continue
            
            # Slide the window to build the differents graphs
            for j in tqdm.tqdm(range(0, n_frames - self.window_size + 1, self.stride)):
                # Build the graph
                node_features, edge_index, frame_mask = self.build_graph(coords_indv[:, j:j+self.window_size, :])

                # Build the data object
                data = Data(x=node_features, edge_index=edge_index, y=file, frame_mask=frame_mask)
                self.dataset.append(data)
            
    def load_behaviour(self, file):
        ''' Function that loads the behaviour from a csv file.

            Args:
                file (str): The csv file to load.

            Returns:
                behaviour (torch.Tensor): The behaviour as a tensor.'''
        
        behaviour = pd.read_csv(file, header=None)
        


