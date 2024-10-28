#### Pre-prcessing functions for the data ####
import dataloader as dl

def preprocess_data(path):
    """
    Preprocess the data
    """
    # Load the data
    data = dl.load_data(path)
    coords = data.coords
