# DATALOADER CLASS to handle the data loading and preprocessing
# We load the .h5 files with the trajectories of DeepLabCut and preprocess them to build the graps
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import time
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

def reload_module():
    importlib.reload(DataDLC)

class DLCDataLoader:
    ''' The DataLoader class for the DeepLabCut data. It loads the data from the .h5 files and preprocesses it to build the graphs. '''
    
    def __init__(self, root, load_dataset = False, batch_size = 1, num_workers = 1, device = 'cpu', window_size=None, stride=None, build_graph=False, behaviour = None):
        ''' Constructor of the DataLoader class. It loads the data from the .h5 files and preprocesses it to build the graphs.

            Args:
                root (str): The root directory of the .h5 files.
                load_dataset (bool): If True, the dataset is loaded from a .pkl file.
                batch_size (int): The batch size.
                num_workers (int): The number of workers for the DataLoader.
                device (torch.device): The device to load the data.
                window_size (int): The window size for the temporal graph.
                stride (int): The stride for the temporal graph.
                spatio_temporal_adj (MultiIndex): The spatio-temporal adjacency matrix.
                build_graph (bool): If True, the graph is built from the coordinates of the individuals
                behavoiur (str): The behaviour to load. '''

        self.root = root
        self.batch_size = batch_size # Batch size
        self.num_workers = num_workers # Number of workers for the DataLoader
        self.device = device # Device to load the data

        self.behaviour = behaviour # Behaviour to load

        # if load_dataset, load the dataset from the .pkl file
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
            self.files.sort(key=lambda x: int(x.split('DLC')[0].split('_')[3]))
            #self.files.sort(key=lambda x: int(x.split('DLC')[0].split('_')[2].split(' ')[1]))
            #self.files.sort()
            self.n_files = len(self.files) # Number of files, i.e. number of spatio-temporal graphs
            self.data_list = []
            #self.behaviour = []
        
            print(f"Loading data from {root}, where we have {self.n_files} files")
            self.load_data_3()	# Load the data
            #self.load_data()
            print(f"Number of files: {self.n_files}")

        #super(DLCDataLoader, self).__init__(self.data_list, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
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
        print(f"Batch size: {self.batch_size}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Device: {self.device}")
        print(f"Window size: {self.window_size}")
        print(f"Stride: {self.stride}")
        print(f"Build graph: {self.build_graph}")

    
    def load_data_2(self):
        '''
        Function that loads the data from the .h5 files and preprocesses it to build the graphs.
        It uses the DataDLC class to load the data. 
        '''                
        
        for i, file in enumerate(self.files):#tqdm.tqdm(enumerate(self.files)):
            # Load the data
            print(f"Loading file {file}", end='\r')
            # Name of the test
            name_file = file.split('DLC')[0]
            # Load the behaviour
            # See if there's a behaviour file
            
            if os.path.exists(os.path.join(self.root, name_file + '.csv')):
                behaviour = self.load_behaviour(name_file + '.csv')
                #behaviour = torch.tensor(behaviour.values, dtype=torch.int64)
            else:
                behaviour = None
                print(f"No behaviour file for {name_file}")

            # Load the data
            dlc = DataDLC.DataDLC(os.path.join(self.root, file))
            # TO DO dlc.preprocess()

            dlc.drop_tail_bodyparts()
            
            if self.buid_graph:
                # Build the graph
                t0 = time.time()
                node_features, edge_index, frame_mask = self.build_graph_3(data_dlc=dlc, window_size= self.window_size, stride = self.stride)
                print(f"Graph built in {time.time() - t0} s")
                data = []
                for i in range(len(node_features)):
                    frames = frame_mask[i].unique().tolist()
                    behaviour_window = behaviour.loc[frames]
                    data.append(Data(x=node_features[i], edge_index=edge_index[i], y=behaviour_window, frame_mask=frame_mask[i], file=file))
                self.data_list.append(data)
            else:
                self.data_list.append((dlc.coords, behaviour))
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
            else:
                behaviour = None
                print(f"No behaviour file for {name_file}")
            if self.behaviour is not None:
                behaviour = behaviour[self.behaviour]

            data_dlc = DataDLC.DataDLC(os.path.join(self.root, file))

            #data_dlc.cast_boudaries()

            data_dlc.drop_tail_bodyparts()

            # Numpy is faster than pandas
            coords = data_dlc.coords.to_numpy()
            
            # Reshape the coordinates to have the same shape as the original data (n_frames, n_individuals, n_body_parts, 3)
            coords = coords.reshape((coords.shape[0], data_dlc.n_individuals, data_dlc.n_body_parts, 3))

            if self.buid_graph:

                if self.window_size is None:
                    # Build the graph
                    node_features, edge_index, frame_mask = self.build_graph_4(coords)
                    # Build the data object

                    data = Data(x=node_features, edge_index=edge_index, file=file, frame_mask=frame_mask, behaviour= torch.tensor(behaviour, dtype=torch.long))
                    self.data_list.append(data)
                    continue

                # Slide the window to build the differents graphs
                for j in tqdm.tqdm(range(0, data_dlc.n_frames - self.window_size + 1, self.stride)):
                    # Only cae about the central frame of the window
                    behaviour_window = behaviour.iloc[j+self.window_size//2]

                    # Build the graph
                    node_features, edge_index, frame_mask = self.build_graph_4(coords[j:j+self.window_size])
                    frame_mask += j

                    # Build the data object
                    data = Data(x=node_features, edge_index=edge_index, file=file, frame_mask=frame_mask, behaviour=torch.tensor(behaviour_window, dtype=torch.long))
                    self.data_list.append(data)
            else:
                self.data_list.append((data_dlc.coords, behaviour))

    def get_dataloader(self):
        ''' Function that returns the DataLoader object. '''
        return DataLoader(self.data_list, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def load_data(self):
    
        print(f"We have {self.n_files} files")
        for i, file in enumerate(self.files):
            
            print(f"Loading file {file}")
            name_file = file.split('DLC')[0]
            if os.path.exists(os.path.join(self.root, name_file + '.csv')):
                behaviour = self.load_behaviour(name_file + '.csv')
            else:
                behaviour = None
                print(f"No behaviour file for {name_file}")

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
                t0 = time.time()
                data = Data(x=node_features, edge_index=edge_index, file=file, frame_mask=frame_mask, behaviour=[behaviour], Name = name_file)
                print(f"Graph built in {time.time() - t0} s")
                self.data_list.append(data)
                continue
            
            # Slide the window to build the differents graphs
            for j in tqdm.tqdm(range(0, n_frames - self.window_size + 1, self.stride)):

                # Behaviour in the window
                behaviour_window = behaviour[j:j+self.window_size]

                # Build the graph
                node_features, edge_index, frame_mask = self.build_graph(coords_indv[:, j:j+self.window_size, :])

                # Build the data object
                data = Data(x=node_features, edge_index=edge_index, file=file, frame_mask=frame_mask, behaviour=behaviour_window, Name = name_file)
                self.data_list.append(data)

    def build_graph_4(self, coords):
        ''' Function that builds the graph from the coordinates of the individuals. 

            The nodes feqtures are the coordinates of the individuals, with the likelihood of the body parts, as well as the individual index.

            The graph will have edges between all the nodes of the same individual in the same frame,
            the nose of the individuals in the same frame and the tail of the individuals in the same frame.
            Also we have edges between the nodes of the same body part accross adjecent frames.

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
        n_edges = n_individuals * n_body_parts**2 * n_frames + n_individuals * n_body_parts * (n_frames - 1) + n_individuals*(n_individuals-1)*3
        # Initialize the node features
        node_features = torch.zeros(n_nodes, 4, dtype=torch.float32)
        # Initialize the edge index
        #edge_index = torch.zeros(2, n_edges, dtype=int)
        edge_list = []

        # Nose index, Tail index
        idx_nose = 0
        idx_tail = n_body_parts - 2 # The last body part is the center of mass, the tail is the one before (if all the other body parts where dropped)

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
                    # edge_index[0, edge] = node
                    # edge_index[1, edge] = node
                    edge += 1

                    # Edges between the nodes of the same individual in the same frame, only the nodes already created
                    for l in range(0, j):
                        edge_list.append((node, 
                                          i * n_body_parts * n_frames + l * n_frames + k))
                        # edge_index[0, edge] = node
                        # edge_index[1, edge] = i * n_body_parts * n_frames + l * n_frames + k
                        edge += 1
                    # Edges between the nodes of the same body part accross adjecent frames
                    if k < n_frames - 1:

                        edge_list.append((node,
                                           node + 1))
                        # edge_index[0, edge] = node
                        # edge_index[1, edge] = node + 1
                        
                        edge += 1

                    if j == idx_nose:
                        # Nose-Nose, Nose-Tail, Tail-Tail edges between individuals
                        for i2 in range(0, i):
                            # Nose
                            edge_list.append((i * n_body_parts * n_frames + idx_nose * n_frames, 
                                              i2 * n_body_parts * n_frames + idx_nose * n_frames))
                            # edge_index[0, edge] = i * n_body_parts * n_frames + idx_nose * n_frames
                            # edge_index[1, edge] = i2 * n_body_parts * n_frames + idx_nose * n_frames
                            edge += 1
                    
                    if j == idx_tail:
                        # Tail
                        for i2 in range(0, i):
                            edge_list.append((i * n_body_parts * n_frames + idx_tail * n_frames + k,
                                               i2 * n_body_parts * n_frames + idx_tail * n_frames + k ))
                            # edge_index[0, edge] = i * n_body_parts * n_frames + idx_tail * n_frames + k
                            # edge_index[1, edge] = i2 * n_body_parts * n_frames + idx_tail * n_frames + k
                            edge += 1
                        # Nose-Tail
                        for i2 in range(0, i):
                            edge_list.append((i * n_body_parts * n_frames + idx_nose * n_frames + k,
                                               i2 * n_body_parts * n_frames + idx_tail * n_frames + k))
                            # edge_index[0, edge] = i * n_body_parts * n_frames + idx_nose * n_frames + k
                            # edge_index[1, edge] = i2 * n_body_parts * n_frames + idx_tail * n_frames + k
                            edge += 1

                            edge_list.append((i2 * n_body_parts * n_frames + idx_nose * n_frames + k,
                                               i * n_body_parts * n_frames + idx_tail * n_frames + k))
                            # edge_index[0, edge] = i2 * n_body_parts * n_frames + idx_nose * n_frames + k
                            # edge_index[1, edge] = i * n_body_parts * n_frames + idx_tail * n_frames + k
                            edge += 1

        edge_index = torch.tensor(edge_list, dtype=int).T

        return node_features, edge_index, frame_mask

                



                    


    def build_graph(self, coords_indv):
        ''' Function that builds the graph from the coordinates of the individuals.
            First dimension of the input is the number of individuals.
            Second dimension is the number of frames.
            Third dimension is the number of body parts mutiplyed by 3 (x, y, likelihood).

            We have edges between all the nodes of the same individual in the same frame, 
            the nose of the individuals in the same frame and the tail of the individuals in the same frame.
            Also we have edges between the nodes of the same body part accross adjecent frames.

            Args:
                coords_indv (list): List of the coordinates of the individuals.

            Returns:
                node_features (torch.Tensor): The node features of the graph.
                edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].'''
        
        # Get the number of individuals
        n_individuals = coords_indv.shape[0]
        # Get the number of frames
        n_frames = coords_indv.shape[1]
        # Get the number of body parts
        n_body_parts = coords_indv.shape[2] // 3
        # Get the number of nodes
        n_nodes = n_individuals * n_body_parts * n_frames
        # node-level frame mask grey encoding
        frame_mask = torch.zeros(n_nodes, dtype=torch.int64)
        # Get the number of edges
        # Edges between the nodes of the same individual in the same frame + edges between same body parts in adjecent frames 
        n_edges = n_individuals * n_body_parts**2 * n_frames + n_individuals * n_body_parts * (n_frames - 1) 
        # Initialize the node features
        node_features = torch.zeros(n_nodes, 3)
        # Initialize the edge index
        edge_index = torch.zeros(2, n_edges, dtype=int)
        # Fill the node features
        for i in range(n_individuals):
            for j in range(n_body_parts):
                for k in range(n_frames):
                    node = i * n_body_parts * n_frames + j * n_frames + k
                    node_features[node, :] = coords_indv[i, k, j*3:j*3+3]
                    frame_mask[node] = k

        # Fill the edge index   
        edge = 0
        for i in range(n_individuals):
            for j in range(n_body_parts):
                for k in range(n_frames):
                    node = i * n_body_parts * n_frames + j * n_frames + k

                    # Edges between the nodes of the same individual in the same frame
                    for l in range(n_body_parts):
                        edge_index[0, edge] = node
                        edge_index[1, edge] = i * n_body_parts * n_frames + l * n_frames + k
                        edge += 1
                    # Edges between the nodes of the same body part accross adjecent frames
                    if k < n_frames - 1:
                        edge_index[0, edge] = node
                        edge_index[1, edge] = node + 1
                        edge += 1
                        #edge_index[0, edge] = node + 1
                        #edge_index[1, edge] = node
                        #edge += 1
        # Nose-Nose, Nose-Tail, Tail-Tail edges between individuals
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i == j:
                    continue
                # Nose
                edge_index[0, edge] = i * n_body_parts * n_frames + 0 * n_frames + 0
                edge_index[1, edge] = j * n_body_parts * n_frames + 0 * n_frames + 0
                edge += 1
                # Tail_base
                edge_index[0, edge] = i * n_body_parts * n_frames + (n_body_parts-1) * n_frames + 0
                edge_index[1, edge] = j * n_body_parts * n_frames + (n_body_parts-1) * n_frames + 0
                edge += 1
                # Nose-Tail_base
                edge_index[0, edge] = i * n_body_parts * n_frames + 0 * n_frames + 0
                edge_index[1, edge] = j * n_body_parts * n_frames + (n_body_parts-1) * n_frames + 0
                edge += 1
        return node_features, edge_index, frame_mask
    

    ####### TO OPTIMISE, IT TAKES TOO LONG, PROBABLY BECAUSE OF THE DATA CLASS MANEGMENT, IT TAKES TO LONG TO ACCESS THE DATA #######
    def build_graph_2(self, data_dlc: DataDLC.DataDLC, spatio_temporal_adj: pd.MultiIndex, window_size: int, stride: int) -> [torch.Tensor, torch.LongTensor, torch.Tensor]: 
        ''' 
        Function that builds the graph from an instance of DataDLC class.

        Args:
            data_dlc (DataDLC): The instance of the DataDLC class.
            spatio_temporal_adj (pd.MultiIndex): The spatio-temporal adjacency matrix.
            window_size (int): The window size for the temporal graph.
            stride (int): The stride for the temporal graph.

        Returns:
            node_features (torch.Tensor): The node features of the graph.
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
            frame_mask (torch.Tensor): The frame mask of the graph.
        '''
        # Get the number of Nan values
        n_nan = data_dlc.coords.isnull().sum().sum()
        # Get the number of nodes
        n_nodes = data_dlc.n_individuals * data_dlc.n_body_parts * data_dlc.n_frames - n_nan
        # Frame mask
        frame_mask = []
        # Initialize the node features as a list
        node_features = []
        # Initialize the edge index as a list
        edge_index = []
        # Build prescence mask of the body parts, and graph index_node
        node_index_dict = data_dlc.coords.copy() # Build a multiindex dataframe with the data
        node_index_dict = node_index_dict.xs('x', level=2, axis=1) # keep only one column per body part
        node_index_dict = node_index_dict.applymap(lambda x: np.nan) # Fill with NaNs all data
        
        # Fill the node features and the edge index
        for frame in tqdm.tqdm(range(data_dlc.n_frames)):
            for ind_idx, individual in enumerate(data_dlc.individuals):
                for body_part_idx, body_part in enumerate(data_dlc.body_parts):
                    if np.isnan(data_dlc.coords[individual].loc[frame][body_part]).any():
                        continue
                    node_features.append(data_dlc.coords[individual].loc[frame][body_part].values)
                    frame_mask.append(frame)

                    current_node = len(node_features) - 1
                    node_index_dict.loc[frame, (individual, body_part)] = current_node # We save the node_index corresponding to the body part
                    # Add all Edges possible for the current node (with all the nodes that were already created)
                    # First we add the edges between the nodes of the same individual in the same frame
                    for body_part_2 in data_dlc.body_parts[:body_part_idx]: # We only add the edges with the body parts that were included in the graph
                        if np.isnan(node_index_dict.loc[frame, (individual, body_part_2)]):
                            continue
                        node_2 = node_index_dict.loc[frame, (individual, body_part_2)]
                        edge_index.append([current_node, node_2])
                        edge_index.append([node_2, current_node])
                        
                    # Then we add the edges between the nodes of the same body part in the previous frame
                    if frame>0 and not np.isnan(node_index_dict.loc[frame-1, (individual, body_part)]):
                        node_2 = node_index_dict.loc[frame-1, (individual, body_part)]
                        edge_index.append([current_node, node_2])
                        edge_index.append([node_2, current_node])

                # Nose-Nose, Nose-Tail, Tail-Tail edges between individuals
                for ind_idx_2, individual_2 in enumerate(data_dlc.individuals[:ind_idx]): # We only add the edges with the individuals that were included in the graph
                    # Nose
                    if not np.isnan(node_index_dict.loc[frame, (individual, 'Nose')]): 
                        node_1 = node_index_dict.loc[frame, (individual, 'Nose')]
                        if not np.isnan(node_index_dict.loc[frame, (individual_2, 'Nose')]): # Nose-Nose
                            node_2 = node_index_dict.loc[frame, (individual_2, 'Nose')]
                            edge_index.append([node_1, node_2])
                            edge_index.append([node_2, node_1])
                        if not np.isnan(node_index_dict.loc[frame, (individual_2, 'Tail_base')]): # Nose-Tail 
                            node_2 = node_index_dict.loc[frame, (individual_2, 'Tail_base')]
                            edge_index.append([node_1, node_2])
                            edge_index.append([node_2, node_1])
                    # Tail
                    if not np.isnan(node_index_dict.loc[frame, (individual, 'Tail_base')]): 
                        node_1 = node_index_dict.loc[frame, (individual, 'Tail_base')]
                        if not np.isnan(node_index_dict.loc[frame, (individual_2, 'Nose')]): # Tail-Nose
                            node_2 = node_index_dict.loc[frame, (individual_2, 'Nose')]
                            edge_index.append([node_1, node_2])
                            edge_index.append([node_2, node_1])
                        if not np.isnan(node_index_dict.loc[frame, (individual_2, 'Tail_base')]): # Tail-Tail
                            node_2 = node_index_dict.loc[frame, (individual_2, 'Tail_base')]
                            edge_index.append([node_1, node_2])
                            edge_index.append([node_2, node_1])
                        
            

        # Convert the lists to tensors
        # Node features to Tensor
        node_features = torch.tensor(node_features, dtype=torch.float32)
        # Edge index to sparse tensor (COO format)
        edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
        frame_mask = torch.tensor(frame_mask, dtype=torch.int64)

        return node_features, edge_index, frame_mask

    def build_graph_3(self, data_dlc: DataDLC.DataDLC, window_size: int, stride: int) -> [torch.Tensor, torch.LongTensor, torch.Tensor]:
        ''' 
        Function that builds the graph from an instance of DataDLC class.

        Args:
            data_dlc (DataDLC): The instance of the DataDLC class.
            spatial_adj (pd.MultiIndex): The spatial adjacency matrix.
            window_size (int): The window size for the temporal graph.
            stride (int): The stride for the temporal graph.

        Returns:
            node_features_list (List[torch.Tensor]): The node features of the graphs, each element of the list is a tensor of shape [num_nodes(t), num_features].
            edge_index_list (list[LongTensor]): Graph connectivity in COO format with shape [2, num_edges(t)] for each graph.
            frame_mask (list[torch.Tensor]): The frame mask of each node of the graphs.
        '''

        # pad window_size//2 frames at the beginning and end of the video to avoid edge cases.
        #data_dlc.coords = pd.concat([pd.DataFrame(np.nan, index=range(window_size//2), columns=data_dlc.coords.columns), data_dlc.coords, pd.DataFrame(np.nan, index=range(window_size//2), columns=data_dlc.coords.columns)])
        #data_dlc.coords = data_dlc.coords.reset_index(drop=True) # reset index to have a continuous index (REMEMBER THAT WE PADDED THE DATA, SO THE INDEX ARE SHIFTED WRT THE FRAMES)

        # Build the graphs
        node_features = []
        edge_index = []
        
        # Frame mask
        frame_mask_list = []
        # Initialize the node features as a list
        node_features_list = []
        # Initialize the edge index as a list
        edge_index_list = []
        # Build prescence mask of the body parts, and graph index_node
        node_index_dict = data_dlc.coords.copy() # Build a multiindex dataframe with the data
        node_index_dict = node_index_dict.xs('x', level=2, axis=1) # keep only one column per body part
        node_index_dict = node_index_dict.applymap(lambda x: np.nan) # Fill with NaNs all data

        if window_size==-1:
            n_graphs = 1
            window_size = data_dlc.n_frames
            stride = data_dlc.n_frames
        else:
        #    Number of graphs 
            n_graphs = (data_dlc.n_frames - window_size) // stride + 1
        print(f"Number of graphs: {n_graphs} \n ")

        for t in tqdm.tqdm(range(n_graphs)):
            #print(f"Building graph {t}/{n_graphs}", end='\r')
            # Fill the node features and the edge index
            node_features = []
            edge_index = []
            frame_mask = []
            # Build prescence mask of the body parts, and graph index_node
            node_index_dict = data_dlc.coords.iloc[t*stride:(t*stride + window_size)].copy() # Build a multiindex dataframe with the data
            node_index_dict = node_index_dict.xs('x', level=2, axis=1) # keep only one column per body part
            node_index_dict = node_index_dict.applymap(lambda x: np.nan) # Fill with NaNs all data

            for frame in range(t*stride, t*stride + window_size):
                for ind_idx, individual in enumerate(data_dlc.individuals):
                    for body_part_idx, body_part in enumerate(data_dlc.body_parts):
                        if np.isnan(data_dlc.coords[individual].loc[frame][body_part]).any():
                            continue
                        feat_node = data_dlc.coords[individual].loc[frame][body_part].values
                        # Concatenate individual idx
                        feat_node = np.concatenate((feat_node, np.array([ind_idx])))
                        node_features.append(feat_node)
                        frame_mask.append(frame)

                        # Add edges
                        current_node = len(node_features) - 1
                        node_index_dict.loc[frame, (individual, body_part)] = current_node # We save the node_index corresponding to the body part
                        # Add all Edges possible for the current node (with all the nodes that were already created)
                        # First we add the edges between the nodes of the same individual in the same frame
                        for body_part_2 in data_dlc.body_parts[:body_part_idx]: # We only add the edges with the body parts that were included in the graph
                            if np.isnan(node_index_dict.loc[frame, (individual, body_part_2)]):
                                continue
                            node_2 = node_index_dict.loc[frame, (individual, body_part_2)]
                            edge_index.append([current_node, node_2])
                            edge_index.append([node_2, current_node])

                        # Then we add the edges between the nodes of the same body part in the previous frame
                        if frame>t*stride and not np.isnan(node_index_dict.loc[frame-1, (individual, body_part)]):
                            node_2 = node_index_dict.loc[frame-1, (individual, body_part)]
                            edge_index.append([current_node, node_2])
                            edge_index.append([node_2, current_node])

                    # Nose-Nose, Nose-Tail, Tail-Tail edges between individuals
                    for ind_idx_2, individual_2 in enumerate(data_dlc.individuals[:ind_idx]): # We only add the edges with the individuals that were included in the graph
                        # Nose
                        if not np.isnan(node_index_dict.loc[frame, (individual, 'Nose')]):
                            node_1 = node_index_dict.loc[frame, (individual, 'Nose')]
                            if not np.isnan(node_index_dict.loc[frame, (individual_2, 'Nose')]):
                                node_2 = node_index_dict.loc[frame, (individual_2, 'Nose')]
                                edge_index.append([node_1, node_2])
                                edge_index.append([node_2, node_1])
                            if not np.isnan(node_index_dict.loc[frame, (individual_2, 'Tail_base')]):
                                node_2 = node_index_dict.loc[frame, (individual_2, 'Tail_base')]
                                edge_index.append([node_1, node_2])
                                edge_index.append([node_2, node_1])
                        # Tail
                        if not np.isnan(node_index_dict.loc[frame, (individual, 'Tail_base')]):
                            node_1 = node_index_dict.loc[frame, (individual, 'Tail_base')]
                            if not np.isnan(node_index_dict.loc[frame, (individual_2, 'Nose')]):
                                node_2 = node_index_dict.loc[frame, (individual_2, 'Nose')]
                                edge_index.append([node_1, node_2])
                                edge_index.append([node_2, node_1])
                            if not np.isnan(node_index_dict.loc[frame, (individual_2, 'Tail_base')]):
                                node_2 = node_index_dict.loc[frame, (individual_2, 'Tail_base')]
                                edge_index.append([node_1, node_2])
                                edge_index.append([node_2, node_1])

            # Convert the lists to tensors
            # Node features to Tensor
            node_features = torch.tensor(node_features, dtype=torch.float32)
            # Edge index to sparse tensor (COO format)
            edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
            frame_mask = torch.tensor(frame_mask, dtype=torch.int64)

            node_features_list.append(node_features)
            edge_index_list.append(edge_index)
            frame_mask_list.append(frame_mask)

        return node_features_list, edge_index_list, frame_mask_list
            
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



    


        


