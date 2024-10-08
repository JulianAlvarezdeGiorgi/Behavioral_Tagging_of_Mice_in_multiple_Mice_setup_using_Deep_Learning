import models
import torch
import numpy as np
import importlib 
from torch_geometric.data import Data, DataLoader
import tqdm
import pandas as pd
import torch.nn as nn


# Define a Dictionary that contains the models for each behavior
MODELS = {'General_Contact': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Sniffing': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Sniffing_head': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Sniffing_other': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Sniffing_anal': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Poursuit': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Dominance': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Rearing': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean'],
        'Grooming': [models.GATEncoder(nout = 64, nhid=32, attention_heads = 2, n_in = 4, n_layers=4, dropout=0.2), models.ClassificationHead(n_latent=64, nhid = 32, nout = 2), 'mean']
        }

MODELS_PATH = {'General_Contact': r'd:\Backup_mantenimiento_ruche\Data\Checkpoints\new_encoder_no_linearResCon\General_Contacts\checkpoint_epoch_310.pth',
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
    gatencoder = MODELS[behavior][0]
    classifier = MODELS[behavior][1]
    readout = MODELS[behavior][2]

    return models.GraphClassifier(encoder=gatencoder, classifier=classifier, readout=readout)

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

def inference(behaviour, path_to_data, path_to_save):
    ''' This function runs the inference on the specified behavior, and save
        the results in the specified path.
    Args:
        behaviour: str, the behavior on which to run the inference
        path_to_data: str, the path to the dataset to run the inference on
        path_to_save: str, the path where to save the results
    ''' 

    model_path = MODELS_PATH[behaviour] # get the model path
    model = load_model(model_path, DEVICE, behaviour) # load the model
    data = torch.load(path_to_data, map_location=DEVICE) # load the data
    loader = DataLoader(data, batch_size=1, shuffle=False) # create the DataLoader

    outputs = np.zeros((len(loader), 2)) # create the outputs array
    softmax = nn.Softmax(dim=1) # create the softmax function
    for i, batch in enumerate(tqdm.tqdm(loader)):
        outputs[i][0] = int(batch.frame_mask.median().item())
        with torch.no_grad():
            out = model(batch)
            out = softmax(out)
            outputs[i][1] = out.argmax(dim=1).cpu().numpy()

    create_csv_with_output_behaviour(outputs, behaviour, path_to_save) # create the csv file with the results

