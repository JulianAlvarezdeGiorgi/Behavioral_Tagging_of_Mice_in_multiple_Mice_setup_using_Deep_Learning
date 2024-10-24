# DATALOADER CLASS to handle the data loading and preprocessing
# We load the .h5 files with the trajectories of DeepLabCut and preprocess them to build the graps
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import time
import random
#from statsmodels.tsa.arima.model import ARIMA

import h5py
import numpy as np
import os
import torch
#import utils as ut
import pandas as pd
import numpy as np
import tqdm
import cv2
import matplotlib.pyplot as plt
#Import the DataDLC class
import DataDLC
import importlib # to reload the DataDLC class

class DLCDataLoader:
    ''' The DataLoader class for the DeepLabCut data. It loads the data from the .h5 files and preprocesses it to build the graphs. '''
    
    def __init__(self, root, load_dataset = False, window_size=None, stride=None, build_graph=False, behaviour = None, progress_callback = None):
        ''' Constructor of the DataLoader class. It loads the data from the .h5 files and preprocesses it to build the graphs.

            Args:
                root (str): The root directory of the .h5 files.
                load_dataset (bool): If True, the dataset is loaded from a .pkl file.
                window_size (int): The window size for the temporal graph.
                stride (int): The stride for the temporal graph.
                spatio_temporal_adj (MultiIndex): The spatio-temporal adjacency matrix.
                build_graph (bool): If True, the graph is built from the coordinates of the individualss
                behavoiur (str): The behaviour to load. 
                progress_callback (function): The progress callback function (necessary for the GUI).
        '''

        self.root = root
        self.progress_callback = progress_callback # Progress callback function

        self.behaviour = behaviour # Behaviour to load

        # If load_dataset, load the dataset from the .pkl file
        if load_dataset:
            # search for .pkl file
            files = [f for f in os.listdir(root) if f.endswith('.pkl')]
            if len(files) == 0:
                raise ValueError("No .pkl file found in the root directory")
            else:
                self.data_list = torch.load(os.path.join(root, files[0]))
                self.n_files = len(self.data_list)
                # Init the DataLoader
                print(f"Dataset loaded from {os.path.join(root, files[0])}")
        else:
            # Window size must be odd or None
            if window_size is not None and window_size % 2 == 0:
                raise ValueError("Window size must be odd or None")
            self.window_size = window_size
            self.stride = stride # Stride for the temporal graph
            self.buid_graph = build_graph
            
            
            self.files = [f for f in os.listdir(root) if f.endswith('filtered.h5')]
            print(self.files)
            # Order by number of the test
            #self.files.sort(key=lambda x: int(x.split('DLC')[0].split('_')[3]))
            self.n_files = len(self.files) # Number of files
            self.data_list = []
        
            print(f"Loading data from {root}, where we have {self.n_files} files")
            self.load_data_3()	# Load the data
            print(f"Number of files: {self.n_files}")

    def __len__(self):
        ''' Function that returns the number of files. '''
        return self.n_files
    
    def __getitem__(self, idx):
        ''' Function that returns the data at a given index.

            Args:
                idx (int): The index of the data.

            Returns:
                data (Data): The data at the given index.'''
        return self.data_list[idx]
    
    def print_info(self):
        ''' Function that prints the information about the DataLoader. '''
        print(f"Number of files: {self.n_files}")
        print(f"Files are: {self.files}")
        print(f"Device: {self.device}")
        print(f"Window size: {self.window_size}")
        print(f"Stride: {self.stride}")
        print(f"Build graph: {self.build_graph}")

    

    def load_data_3(self):
        '''
        Function that loads the data from the .h5 files and preprocesses it to build the graphs.
        It uses the DataDLC class to load the data. 
        '''                
        print(f"We have {self.n_files} files")
        for i, file in enumerate(self.files):
        
            print(f"Loading file {file}")
            name_file = file.split('DLC')[0]
            if os.path.exists(os.path.join(self.root, name_file + '.csv')):
                behaviour = self.load_behaviour(name_file + '.csv')
                # Drop the first column (frame number)
                # Chec if the column Frames or Frame is present
                if 'Frames' in behaviour.columns:
                    behaviour = behaviour.drop(columns='Frames')

                elif 'frame' in behaviour.columns:
                    behaviour = behaviour.drop(columns='frame')
            else:
                behaviour = None
                print(f"No behaviour file for {name_file}")
            if self.behaviour is not None:
                behaviour = behaviour[self.behaviour]

            data_dlc = DataDLC.DataDLC(os.path.join(self.root, file))

            data_dlc.drop_tail_bodyparts()

            coords = data_dlc.coords.to_numpy()
            # Cast the boundaries
            
            coords = self.cast_boundaries(coords)
            coords = self.normalize_coords(coords)
            
            # Reshape the coordinates to have the same shape as the original data (n_frames, n_individuals, n_body_parts, 3)
            coords = coords.reshape((coords.shape[0], data_dlc.n_individuals, data_dlc.n_body_parts, 3))

            if self.buid_graph:

                if self.window_size is None:
                    # Build the graph
                    node_features, edge_index, frame_mask = self.build_graph_5(coords)
                    # Build the data object

                    data = Data(x=node_features, edge_index=edge_index, file=file, frame_mask=frame_mask, behaviours= torch.tensor(behaviour.values, dtype=torch.long), behaviour_names = behaviour.columns)
                    self.data_list.append(data)
                    continue

                # Slide the window to build the differents graphs
                for j in tqdm.tqdm(range(0, data_dlc.n_frames - self.window_size + 1, self.stride)):
                    # Only cae about the central frame of the window
                    if behaviour is not None:
                        behaviour_window = behaviour.iloc[j+self.window_size//2]
                    # Build the graph
                    node_features, edge_index, frame_mask = self.build_graph_5(coords[j:j+self.window_size])
                    frame_mask += j

                    # Build the data object
                    if behaviour is not None:
                        data = Data(x=node_features, edge_index=edge_index, file=file, frame_mask=frame_mask, behaviour=torch.tensor(behaviour_window.values, dtype=torch.long), behaviour_names = behaviour.columns)
                    else:
                        data = Data(x=node_features, edge_index=edge_index, file=file, frame_mask=frame_mask)
                    self.data_list.append(data)
                    if self.progress_callback:
                        self.progress_callback(j + 1, data_dlc.n_frames - self.window_size + 1)
            else:
                self.data_list.append((data_dlc.coords, behaviour))


                


    def build_graph_5(self, coords) -> (torch.Tensor, torch.LongTensor, torch.Tensor):
        ''' The same implementation logic as build_graph_4 but a more complete graph, edges between nose and all "border" body parts of the other individuals will be included 
            
            Args:
                coords (np.ndarray): The coordinates of the individuals.

            Returns:
                node_features (torch.Tensor): The node features of the graph.
                edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
                frame_mask (torch.Tensor): The frame mask of the graph.
        '''
        # Get the number of individuals
        n_individuals = coords.shape[1]
        # Get the number of frames
        n_frames = coords.shape[0]
        # Get the number of body parts
        n_body_parts = coords.shape[2]
        # Get the number of nodes
        n_nodes = n_individuals * n_body_parts * n_frames
        # node-level frame mask grey encoding
        frame_mask = torch.zeros(n_nodes, dtype=torch.int32)
        # Get the number of edges
        # Edges between the nodes of the same individual in the same frame + edges between same body parts in adjecent frames + nose-tail edges between individuals
        #n_edges = n_individuals * n_body_parts**2 * n_frames + n_individuals * n_body_parts * (n_frames - 1) + n_individuals*(n_individuals-1)*3
        # Edge body parts
        edge_bp = ['Left_ear', 'Right_ear', 'Left_fhip', 'Right_fhip', 'Left_mid', 'Right_mid', 'Left_bhip', 'Right_bhip', 'Tail_base']
        # Index of the edge_bp
        idx_edge_bp = [1, 2, 4, 5, 9, 10, 11, 12, 16]

        # Initialize the node features
        node_features = torch.zeros(n_nodes, 4, dtype=torch.float32)
        # Initialize the edge index
        #edge_index = torch.zeros(2, n_edges, dtype=int)
        edge_list = []

        # Nose index, Tail index
        idx_nose = 0
        edge = 0
        
        # Fill the node features
        for i in range(n_individuals):
            for j in range(n_body_parts):
                for k in range(n_frames):
                    node = i * n_body_parts * n_frames + j * n_frames + k
                    #node_features[node, :3] = torch.from_numpy(coords[k, i, j])
                    node_features[node, :3] = torch.tensor(coords[k, i, j])
                    #node_features[node, :3] = coords[k, i, j]
                    node_features[node, 3] = i
                    frame_mask[node] = k

                    # Self-loops
                    edge_list.append((node, 
                                      node))
                    edge += 1

                    # Edges between the nodes of the same individual in the same frame, only the nodes already created
                    for l in range(0, j):
                        # undirected edges
                        edge_list.append((node, 
                                          i * n_body_parts * n_frames + l * n_frames + k))
                        edge_list.append((i * n_body_parts * n_frames + l * n_frames + k,
                                          node))
                        edge += 1
                    # Edges between the nodes of the same body part accross adjecent frames
                    if k < n_frames - 1:

                        edge_list.append((node,
                                           node + 1))
                        edge_list.append((node + 1,
                                             node))
                        edge += 1

                    if j == idx_nose:
                       for i2 in range(0, n_individuals):
                            if i != i2:
                                # Nose only once because it will be back in the other loop
                                edge_list.append((i * n_body_parts * n_frames + idx_nose * n_frames + k, 
                                                i2 * n_body_parts * n_frames + idx_nose * n_frames + k)) 
                                edge += 1
                                for idx_bp in idx_edge_bp:
                                    edge_list.append((i * n_body_parts * n_frames + idx_nose * n_frames + k,
                                                    i2 * n_body_parts * n_frames + idx_bp * n_frames + k))
                                    edge_list.append((i2 * n_body_parts * n_frames + idx_bp * n_frames + k,
                                                    i * n_body_parts * n_frames + idx_nose * n_frames + k))
                                    edge += 1
                   

        edge_index = torch.tensor(edge_list, dtype=int).T

        return node_features, edge_index, frame_mask
    
    def cast_boundaries(self, coords):
        ''' Cast the boundaries of the coordinates to the boundaries of the image.

            Args:
                coords (np.ndarray): The coordinates of the individuals.

            Returns:
                coords (np.ndarray): The coordinates of the individuals with the boundaries casted.'''
        
        x_lim = [0, 640]
        y_lim = [0, 480]
        # Cast the boundaries
        coords[:, 0::3] = np.clip(coords[:, 0::3], x_lim[0], x_lim[1])
        coords[:, 1::3] = np.clip(coords[:, 1::3], y_lim[0], y_lim[1])

        return coords
            
    def normalize_coords(self, coords):
        ''' Normalize the coordinates of the individuals.

            Args:
                coords (np.ndarray): The coordinates of the individuals.

            Returns:
                coords (np.ndarray): The normalized coordinates of the individuals.'''
        
        # Normalize the coordinates
        coords[:, 0::3] = coords[:, 0::3] / 640
        coords[:, 1::3] = coords[:, 1::3] / 480

        return coords
        
    def load_behaviour(self, file):
        ''' Function that loads the behaviour from a csv file.

            Args:
                file (str): The csv file to load.

            Returns:
                behaviour (torch.Tensor): The behaviour as a tensor.'''
        
        return pd.read_csv(os.path.join(self.root, file))
    
    def save_dataset(self, path = None):
        ''' Function that saves the dataset.

            Args:
                path (str): The path to save the dataset.'''
        
        # If path is missing
        if path is None:
            path = os.path.join(self.root, 'dataset.pkl')
        torch.save(self.data_list, path)

    def preprocess(self):
        ''' Function that preprocesses the data. '''
        pass


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, sequence_length):
        self.graphs = graphs
        self.sequence_length = sequence_length
        self.sequences = self.create_sequences()

    def create_sequences(self):
        sequences = []
        # Create sequences of graphs while maintaining temporal coherence
        for idx in range(len(self.graphs) - self.sequence_length + 1):
            sequence = self.graphs[idx: idx + self.sequence_length]
            central_idx = idx + self.sequence_length // 2
            label = self.graphs[central_idx].behaviour
            sequences.append((sequence, label))
        return sequences

    def shuffle(self):
        random.shuffle(self.sequences)  # Shuffle the sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]   


