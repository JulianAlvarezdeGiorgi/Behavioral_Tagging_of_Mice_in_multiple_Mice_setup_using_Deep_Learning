import models
import torch
import numpy as np
import importlib 
from torch_geometric.data import Data, DataLoader
import tqdm
import pandas as pd
import torch.nn as nn
import utils
import os
import matplotlib.pyplot as plt
import joblib
import dataloader

GAT_MASK = {'General_Contacts': "GAT", 'Sniffing': "GAT", 'Sniffing_head': "Linear", 'Sniffing_body': "Linear", 'Sniffing_anogenital': "Linear", 'Following': "GAT", 'Dominance': "GAT", 'Grooming': "GAT"}


# MODELS = {#'General_Contacts': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
#         #'General_Contacts': [False],
#         #'Sniffing': [ models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
#         #'Sniffing': [False],
#         #'Sniffing_head': [False],
#         #'Sniffing_body': [False],
#         #'Sniffing_anal': [False],
#         # 'Sniffing_head': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
#         # 'Sniffing_other': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
#         # 'Sniffing_anal': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
#         #'Following': [False],  
#         #'Following': [True, models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
#         #'Dominance': [False],
#         #'Grooming': [False],

#         #'Dominance': [True, models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
#         # 'Rearing': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
#         #'Grooming': [True, models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean']
        
#         }

MODELS_PATH = {'General_Contacts': ['models/GATmodels/GeneralContact_checkpoint_epoch_610', 'models/baseline_models/new_dataset/model_gencont.pkl'],
               'Sniffing': ['models/GATmodels/Sniffing_R_checkpoint_epoch_570', 'models/baseline_models/new_dataset/model_sniffR.pkl'],
                'Sniffing_head': [None,'models/baseline_models/new_dataset/model_Shead.pkl'],
                'Sniffing_body': [None,'models/baseline_models/new_dataset/model_Sbody.pkl'],
                'Sniffing_anogenital': [None,'models/baseline_models/new_dataset/model_Sanus.pkl'],
                'Following': ['models/GATmodels/Following_checkpoint_epoch_442', 'models/baseline_models/new_dataset/model_poursuitR.pkl'],
                'Dominance':  ['models/GATmodels/Dominance_epoch_750','models/baseline_models/new_dataset/model_domR.pkl'],
                'Grooming': ['models/GATmodels/Grooming_checkpoint_epoch_960','models/baseline_models/new_dataset/model_groomR.pkl'],}
             
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Function that returns the model based on the behavior
def get_model(behavior, gat = True) -> nn.Module:
    ''' Returns the model based on the behavior.
        Possible behaviors: 'General_Contact', 'Sniffing', 'Sniffing_head', 'Sniffing_other', 'Sniffing_anal', 'Poursuit', 'Dominance', 'Rearing', 'Grooming'
    Parameters:
        - behavior: str, the behavior of the model
        - gat: bool, whether the model is a GAT model or not
    Returns:
        - model: nn.Module, the model
    '''
    if gat:
        gatencoder = models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2)
        classifier =  models.ClassificationHead(n_latent=64, nhid = 32, nout = 2)
        readout = 'mean'
        model = models.GraphClassifier(encoder=gatencoder, classifier=classifier, readout=readout)
    else:
        model = None
    return model

def load_model(model_path, device, behaviour = 'General_Contact', gat = True):
    ''' This function loads a model from a given path and returns it.
    Args:
        model_path: path to the model
        device: device on which the model should be loaded
        behaviour: behaviour of the model
        gat: whether the model is a GAT model or not
    Returns:
        model: the loaded model
    '''
    model = get_model(behaviour, gat) # get the model
    if model is None:
        model = joblib.load(model_path) # load the model
    else:
        checkpoint = torch.load(model_path, map_location=device) # load the model
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device) # send the model to the device
        model.eval() # set the model to evaluation mode
    return model

def create_csv_with_output_behaviour(output, behaviour, path):
    ''' This function creates a csv file with the output of the model for each frame.
    Args:
        output: the output of the model
        behaviour: the behaviour analyzed
        path: the path where the csv file should be saved
    '''
    df = pd.DataFrame(output, columns = ['Frame', behaviour])
    df.to_csv(path, index = False)

def inference(behaviour, data, gat = True, save = False, path_to_save = None, video = None):
    ''' This function runs the inference on the specified behavior, and save
        the results in the specified path.
    Args:
        behaviour: str, the behavior on which to run the inference
        data: list of torch_geometric.data.Data or numpy arrays, the data on which to run the inference
        gat: bool, whether to use the GAT model or not (if False, the model is the Linear model)
        save: bool, whether to save the results or not
        path_to_save: str, the path where to save the results (if save is True)
        video: str, the name of the video (if save is True)
    Returns:
        outputs: pd.DataFrame, the results of the inference
    ''' 
    if gat:
        model_path = MODELS_PATH[behaviour][0] # get the model path
    else:
        model_path = MODELS_PATH[behaviour][1] # get the model path
    model = load_model(model_path, DEVICE, behaviour, gat) # load the model
    if gat:
        loader = DataLoader(data, batch_size=1, shuffle=False) # create the DataLoader


    if behaviour == 'General_Contacts':
        if gat:
            outputs = pd.DataFrame(np.zeros((len(loader), 2)), columns = ['Frame', behaviour]) # create the DataFrame to store the results
            softmax = nn.Softmax(dim=1) # create the softmax function
            print('Running inference on General_Contacts')
            for i, batch in enumerate(tqdm.tqdm(loader)):
                outputs.loc[i, 'Frame'] = int(batch.frame_mask.median().item()) # get the frame
                with torch.no_grad():
                    out = model(batch)
                    out = softmax(out)
                    outputs.loc[i, behaviour] = out.argmax(dim=1).cpu().numpy() # get the prediction

        else:
            outputs = pd.DataFrame(np.zeros((len(data), 2)), columns = ['Frame', behaviour])
            outputs['Frame'] = range(len(data))

            print('Running inference on General_Contacts')
            
            y_pred = model.predict(data)
            outputs[behaviour] = y_pred

    else:
       
        if gat:
            outputs = pd.DataFrame(np.zeros((len(loader), 3)), columns = ['Frame', behaviour + '_R', behaviour + '_V']) # create the DataFrame to store the results
            softmax = nn.Softmax(dim=1)
            print('Running inference on', behaviour + '_R')
            for i, batch in enumerate(tqdm.tqdm(loader)):
                outputs.loc[i, 'Frame'] = int(batch.frame_mask.median().item()) 
                with torch.no_grad():
                    out = model(batch)
                    out = softmax(out)
                    outputs.loc[i, behaviour + '_R'] = out.argmax(dim=1).cpu().numpy()
        
            # Swap identities
            utils.swap_identities(data)
            loader = DataLoader(data, batch_size=1, shuffle=False)
            print('Running inference on', behaviour + '_V')
            for i, batch in enumerate(tqdm.tqdm(loader)):
                with torch.no_grad():
                    out = model(batch)
                    out = softmax(out)
                    outputs.loc[i, behaviour + '_V'] = out.argmax(dim=1).cpu().numpy()

        else:
            outputs = pd.DataFrame(np.zeros((len(data), 3)), columns = ['Frame', behaviour + '_R', behaviour + '_V'])
            outputs['Frame'] = range(len(data))
            coords_R = data.copy()
            coords_ind2 = data[:, data.shape[1]//2:].copy()
            data[:, data.shape[1]//2:] = data[:, :data.shape[1]//2]
            data[:, :data.shape[1]//2] = coords_ind2

            coords_V = data.copy()

            del data

            print('Running inference on', behaviour + '_R')
            
            y_pred_R = model.predict(coords_R)
            outputs[behaviour + '_R'] = y_pred_R

            print('Running inference on', behaviour + '_V')
            y_pred_V = model.predict(coords_V)
            outputs[behaviour + '_V'] = y_pred_V
            
    if save:
        outputs.to_csv(os.path.join(path_to_save, video + '_' + behaviour + '_output.csv'), index = False)
    else:
        return outputs
    

  

def inference_all_behaviors(path_to_data, path_to_save, gat_mask = GAT_MASK):
    ''' This function runs the inference on all behaviors, and save
        the results in the specified path.
    Args:
        path_to_data: str, the path to the dataset to run the inference on, it should be a folder with a .pkl file and the .h5 files
        path_to_save: str, the path where to save the results
    ''' 

    data_coords = dataloader.DLCDataLoader(path_to_data, build_graph=False) # create the DataLoader
    data_graph = dataloader.DLCDataLoader(path_to_data, load_dataset=True) # create the DataLoader
    #torch.load(path_to_data, map_location=DEVICE) # load the data
    
    # Check if there're different videos
    videos_graph = np.unique([data.file for data in data_graph])
    videos_coords = np.unique([data[2] for data in data_coords])

    videos_graph = sorted(videos_graph)
    videos_coords = sorted(videos_coords)

    videos = videos_graph if videos_graph == videos_coords else print('The videos are different in the two datasets', videos_graph, videos_coords)

    del videos_graph, videos_coords

    data_per_video_graph = []
    data_per_video_coords = []

    for video in videos:
        data_per_video_graph.append([d for d in data_graph if d.file == video])
        data_per_video_coords.append([d[0] for d in data_coords.data_list if d[2] == video])
    for i, video in enumerate(videos):
        print('Running inference on video', video)
        outputs = []

        for behaviour in MODELS_PATH.keys():
            if gat_mask[behaviour]:
                outputs.append(inference(behaviour, data_per_video_graph[i], save = False, gat = True))
            else:
                outputs.append(inference(behaviour, data_per_video_coords[i][0], save = False, gat = False))
        
        # Concatenate the outputs using the column 'frame' as index
        outputs = [output.set_index('Frame') for output in outputs] # Set the column 'Frame' as index
        
        # Concatenate the outputs
        output = pd.concat(outputs, axis=1) 

        # Sort by frame
        output.sort_values(by = 'Frame', inplace = True)

        # Fill the missing values with 0
        output.fillna(0, inplace = True)

        # Save the outputs
        output.to_csv(os.path.join(path_to_save, video + '_output.csv'))
        
def get_number_of_occurrences(data):
    ''' This function returns the number of occurrences of a behavior in the data. i.e. the number of times a 0 is followed by a 1. '''
    count = 0
    for i in range(len(data)-1):
        if data[i] == 0 and data[i+1] == 1:
            count += 1
    return count

def distribution_of_ocurrencies(data, column, num_interv = 6):
    ''' This function computes the distribution of the number of occurrences of a behavior per decil. 
    
    Args:
    
    data: pd.DataFrame, the data
    column: str, the column of the data to analyze
    num_interv: int, the number of intervals to consider'''
    # Get the length of the data
    length = len(data)
    # Get the interval
    interval = length // num_interv
    # Get the distribution
    distribution = []
    for i in range(num_interv):
        distribution.append(data[column][i*interval:(i+1)*interval].sum())
    return distribution

def plot_distribution(distribution, column, path, video_name):
    ''' This function plots the distribution of the number of occurrences of a behavior per decil. '''
    plt.figure()
    plt.bar(range(6), distribution)
    plt.xlabel('Decil')
    plt.ylabel('Number of occurrences')
    plt.title('Distribution of the number of occurrences of ' + column + ' for video ' + video_name)
    plt.savefig(path)
    plt.close()


def get_statistics(path_to_files, num_intervals = 6):
    ''' This function computes the statistics of the model outputs per video and save them in a csv file. 
    
    Args:
        path_to_files: str, the path to the folder containing the csv files
        num_intervals: int, the number of intervals to consider for the distribution plots
        '''

    # Get the list of csv files
    files = [f for f in os.listdir(path_to_files) if f.endswith('.csv')]

    # Create a DataFrame to store the statistics
    statistics = pd.DataFrame(columns = ['video', 'behavior', 'latancy', 'duration (s)', 'duration (frames)', 'number_of_occurrences'])

    # Make a folder for each video
    for file in files:
        video = file.split('_output')[0]
        if not os.path.exists(os.path.join(path_to_files, video)):
            os.makedirs(os.path.join(path_to_files, video))

    for file in files:
        # Load the data
        data = pd.read_csv(os.path.join(path_to_files, file))
        video = file.split('_output')[0]
        statistics_per_video = pd.DataFrame(columns = ['video', 'behavior', 'latancy', 'duration (s)', 'duration (frames)', 'number_of_occurrences'])
        distribution = pd.DataFrame(columns = ['Behavior', 'interval 0', 'interval 1', 'interval 2', 'intarval 3', 'interval 4', 'interval 5'])
        for column in data.columns[1:]:
            # Get the statistics
            latancy = data[column].idxmax()
            duration = data[column].sum()
            number_of_occurrences = get_number_of_occurrences(data[column])

            # Append the statistics to the DataFrame
            new_row = pd.DataFrame({'video': [video], 'behavior': [column], 'latancy': [latancy], 'duration (s)': [duration / 15 ], 'duration (frames)': [duration], 'number_of_occurrences': [number_of_occurrences]})
            #statistics = pd.concat([statistics, new_row], ignore_index=True)
            statistics_per_video = pd.concat([statistics_per_video, new_row], ignore_index=True)
        
            decils_dist = distribution_of_ocurrencies(data, column)
            dist = pd.DataFrame({'Behavior': [column], 'interval 0': [decils_dist[0]], 'interval 1': [decils_dist[1]], 'interval 2': [decils_dist[2]], 'interval 3': [decils_dist[3]], 'interval 4': [decils_dist[4]], 'interval 5': [decils_dist[5]]})
            distribution = pd.concat([distribution, dist], ignore_index=True)

            # Build images with the distribution 
            plot_distribution(decils_dist, column, os.path.join(path_to_files, video, column + '_distribution.png'), video)
        
        # Save the distribution
        distribution.to_csv(os.path.join(path_to_files, video, 'distribution.csv'), index = False, sep=';')
        
        statistics = pd.concat([statistics, statistics_per_video], ignore_index=True)

        # Save the statistics per video
        statistics_per_video.to_csv(os.path.join(path_to_files, video, 'statistics.csv'), index = False, sep=';')               


    # Save the statistics
    statistics.to_csv(os.path.join(path_to_files, 'statistics.csv'), index = False, sep=';')

    return statistics
