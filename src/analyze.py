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

# Define a Dictionary that contains the models for each behavior
MODELS = {'General_Contacts': [True,  models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Sniffing': [True, models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        # 'Sniffing_head': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        # 'Sniffing_other': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        # 'Sniffing_anal': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Following': [False]  
                      #[True, models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        # 'Dominance': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        # 'Rearing': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        #'Grooming': [True, models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean']
        
        }

MODELS_PATH = {'General_Contacts': r'C:\Users\jalvarez\Documents\Code\GitHubCOde\Behavioral_Tagging_of_Mice_in_multiple_Mice_dataset_using_Deep_Learning\models\GATmodels\GeneralContact_checkpoint_epoch_610',
               'Sniffing': r'C:\Users\jalvarez\Documents\Code\GitHubCOde\Behavioral_Tagging_of_Mice_in_multiple_Mice_dataset_using_Deep_Learning\models\GATmodels\Sniffing_R_checkpoint_epoch_570',
               'Following': r'C:\Users\jalvarez\Documents\Code\GitHubCOde\Behavioral_Tagging_of_Mice_in_multiple_Mice_dataset_using_Deep_Learning\models\GATmodels\Following_checkpoint_epoch_442',
                'Grooming': r'C:\Users\jalvarez\Documents\Code\GitHubCOde\Behavioral_Tagging_of_Mice_in_multiple_Mice_dataset_using_Deep_Learning\models\GATmodels\Grooming_checkpoint_epoch_960'
}


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Function that returns the model based on the behavior
def get_model(behavior) -> nn.Module:
    ''' Returns the model based on the behavior.
        Possible behaviors: 'General_Contact', 'Sniffing', 'Sniffing_head', 'Sniffing_other', 'Sniffing_anal', 'Poursuit', 'Dominance', 'Rearing', 'Grooming'
    Parameters:
        - behavior: str, the behavior of the model
    Returns:
        - model: nn.Module, the model
    '''
    if MODELS[behavior][0]:
        gatencoder = MODELS[behavior][1]
        classifier = MODELS[behavior][2]
        readout = MODELS[behavior][3]
        model = models.GraphClassifier(encoder=gatencoder, classifier=classifier, readout=readout)
    else:
        model = None
    return model

def load_model(model_path, device, behaviour = 'General_Contact'):
    ''' This function loads a model from a given path and returns it.
    Args:
        model_path: path to the model
        device: device on which the model should be loaded
        behaviour: behaviour of the model
    Returns:
        model: the loaded model
    '''
    model = get_model(behaviour) # get the model
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
    df = pd.DataFrame(output, columns = ['frame', behaviour])
    df.to_csv(path, index = False)

def inference(behaviour, data, save = False, path_to_save = None):
    ''' This function runs the inference on the specified behavior, and save
        the results in the specified path.
    Args:
        behaviour: str, the behavior on which to run the inference
        data: list of torch_geometric.data.Data, the data to run the inference on.
        save: bool, whether to save the results or not
        path_to_save: str, the path where to save the results (if save is True)
    Returns:
        outputs: pd.DataFrame, the results of the inference
    ''' 

    model_path = MODELS_PATH[behaviour] # get the model path
    model = load_model(model_path, DEVICE, behaviour) # load the model
    loader = DataLoader(data, batch_size=1, shuffle=False) # create the DataLoader

    if behaviour == 'General_Contacts':
        outputs = pd.DataFrame(np.zeros((len(loader), 2)), columns = ['frame', behaviour]) # create the DataFrame to store the results
        softmax = nn.Softmax(dim=1) # create the softmax function
        print('Running inference on General_Contacts')
        for i, batch in enumerate(tqdm.tqdm(loader)):
            outputs.loc[i, 'frame'] = int(batch.frame_mask.median().item()) # get the frame
            with torch.no_grad():
                out = model(batch)
                out = softmax(out)
                outputs.loc[i, behaviour] = out.argmax(dim=1).cpu().numpy() # get the prediction
    else:
        outputs = pd.DataFrame(np.zeros((len(loader), 3)), columns = ['frame', behaviour + '_R', behaviour + '_V']) # create the DataFrame to store the results
        softmax = nn.Softmax(dim=1)
        print('Running inference on', behaviour + '_R')
        for i, batch in enumerate(tqdm.tqdm(loader)):
            outputs.loc[i, 'frame'] = int(batch.frame_mask.median().item()) 
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
    if save:
        outputs.to_csv(path_to_save, index = False)
    else:
        return outputs

def inference_all_behaviors(path_to_data, path_to_save):
    ''' This function runs the inference on all behaviors, and save
        the results in the specified path.
    Args:
        path_to_data: str, the path to the dataset to run the inference on
        path_to_save: str, the path where to save the results
    ''' 
    data = torch.load(path_to_data, map_location=DEVICE) # load the data
    # Check if there're different videos
    videos = np.unique([data.file for data in data])
    data_per_video = []
    for video in videos:
        data_per_video.append([d for d in data if d.file == video])

    for i, data in enumerate(data_per_video):
        video_name = videos[i].split('DLC')[0]
        print('Running inference on video', video_name)
        outputs = []
        for behaviour in MODELS_PATH.keys():
            outputs.append(inference(behaviour, data, save = False))
        # Concatenate the outputs using the column 'frame' as index
        # Set the column 'frame' as index
        outputs = [output.set_index('frame') for output in outputs]
        # discard the column 'frame' from all the outputs except the first one
        #outputs = [output.drop(columns = ['frame']) for output in outputs[1:]]
        # Concatenate the outputs
        outputs = pd.concat(outputs, axis=1)
        # Save the outputs
        outputs.to_csv(os.path.join(path_to_save, video_name + '_output.csv'))

def get_number_of_occurrences(data):
    ''' This function returns the number of occurrences of a behavior in the data. i.e. the number of times a 0 is followed by a 1. '''
    count = 0
    for i in range(len(data)-1):
        if data[i] == 0 and data[i+1] == 1:
            count += 1
    return count

def distribution_of_ocurrencies_per_decil(data, column):
    ''' This function computes the distribution of the number of occurrences of a behavior per decil. '''
    # Get the length of the data
    length = len(data)
    # Get the decil
    decil = length // 10
    # Get the distribution
    distribution = []
    for i in range(10):
        distribution.append(data[column][i*decil:(i+1)*decil].sum())
    return distribution

def plot_distribution(distribution, column, path, video_name):
    ''' This function plots the distribution of the number of occurrences of a behavior per decil. '''
    plt.figure()
    plt.bar(range(10), distribution)
    plt.xlabel('Decil')
    plt.ylabel('Number of occurrences')
    plt.title('Distribution of the number of occurrences of ' + column + ' for video ' + video_name)
    plt.savefig(path)
    plt.close()


def get_statistics(path_to_files):
    ''' This function computes the statistics of the model outputs per video and save them in a csv file. '''

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
        distribution = pd.DataFrame(columns = ['Behavior', 'decil 0', 'decil 1', 'decil 2', 'decil 3', 'decil 4', 'decil 5', 'decil 6', 'decil 7', 'decil 8', 'decil 9'])
        for column in data.columns[1:]:
            # Get the statistics
            latancy = data[column].idxmax()
            duration = data[column].sum()
            number_of_occurrences = get_number_of_occurrences(data[column])

            # Append the statistics to the DataFrame
            new_row = pd.DataFrame({'video': [video], 'behavior': [column], 'latancy': [latancy], 'duration (s)': [duration / 15 ], 'duration (frames)': [duration], 'number_of_occurrences': [number_of_occurrences]})
            #statistics = pd.concat([statistics, new_row], ignore_index=True)
            statistics_per_video = pd.concat([statistics_per_video, new_row], ignore_index=True)
        
            decils_dist = distribution_of_ocurrencies_per_decil(data, column)
            dist = pd.DataFrame({'Behavior': [column], 'decil 0': [decils_dist[0]], 'decil 1': [decils_dist[1]], 'decil 2': [decils_dist[2]], 'decil 3': [decils_dist[3]], 'decil 4': [decils_dist[4]], 'decil 5': [decils_dist[5]], 'decil 6': [decils_dist[6]], 'decil 7': [decils_dist[7]], 'decil 8': [decils_dist[8]], 'decil 9': [decils_dist[9]]})
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
