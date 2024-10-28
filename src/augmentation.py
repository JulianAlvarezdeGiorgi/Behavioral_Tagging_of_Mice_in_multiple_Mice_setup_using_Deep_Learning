import numpy as np
import torch

def merge_symetric_behaviours(indx_behaviour1, indx_behaviour2, dataset, device= 'cpu'):

    """
    Merge two symetric behaviours in the dataset.
    For example, if the behaviour1 is 'Sniffing_Resident' and behaviour2 is 'Sniffing_Visitor', then the function
    will swap identities in the dataset, and add the events of 'Sniffing_Visitor' to 'Sniffing_Resident', merging them into one behaviour.
    """
    # Get index of the two behaviours


    indices_beh1 = []
    indices_beh2 = []

    for i in range(len(dataset)):
        if dataset[i].behaviour[indx_behaviour1] == 1:
            indices_beh1.append(i)
        if dataset[i].behaviour[indx_behaviour2] == 1:
            indices_beh2.append(i)

    for indx2 in indices_beh2:
        if dataset[indx2].behaviour[indx_behaviour1] == 1:
            # We create a new sample, both behaviours are active
            new_sample = dataset[indx2].clone()
            # Set all behaviours to 0
            new_sample.behaviour = torch.zeros(new_sample.behaviour.shape, dtype=torch.long, device=device)
            new_sample.behaviour[indx_behaviour1] = torch.tensor(1, dtype=torch.long, device=device)
            # Replace 0 by 1, and 1 by 0
            new_sample.x[:,3][new_sample.x[:,3] == 0] = torch.tensor(2)
            new_sample.x[:,3][new_sample.x[:,3] == 1] = torch.tensor(0)
            new_sample.x[:,3][new_sample.x[:,3] == 2] = torch.tensor(1)
            # new sample as a Dataset object
            
            dataset.append(new_sample)
        else:
            # We leave the sample but swap the identity of the individuals and activate the first behaviour
            dataset[indx2].behaviour[indx_behaviour1] = torch.tensor(1, dtype=torch.long)
            dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 0] = torch.tensor(2)
            dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 1] = torch.tensor(0)
            dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 2] = torch.tensor(1)
   
    return

def merge_symetric_behaviours_version2(indx_behaviour1, indx_behaviour2, dataset, device= 'cpu'):

    """
    Merge two symetric behaviours in the dataset.
    For example, if the behaviour1 is 'Sniffing_Resident' and behaviour2 is 'Sniffing_Visitor', then the function
    will swap identities in the dataset, and add the events of 'Sniffing_Visitor' to 'Sniffing_Resident', merging them into one behaviour.

    This function is the same as the previous, except that it does create new samples for all the behaviors occuring in the second individual. 
    This is because I hypothesize that it is important to keep information about the same behaviour in the second individual in order to make the moedel
    learn the difference between the two individuals.
    """
    # Get index of the two behaviours


    indices_beh1 = []
    indices_beh2 = []

    for i in range(len(dataset)):
        if dataset[i].behaviour[indx_behaviour1] == 1:
            indices_beh1.append(i)
        if dataset[i].behaviour[indx_behaviour2] == 1:
            indices_beh2.append(i)

    for indx2 in indices_beh2:
        # if dataset[indx2].behaviour[indx_behaviour1] == 1:
        # We create a new sample, both behaviours are active
        new_sample = dataset[indx2].clone()
        # Set all behaviours to 0
        new_sample.behaviour = torch.zeros(new_sample.behaviour.shape, dtype=torch.long, device=device)
        new_sample.behaviour[indx_behaviour1] = torch.tensor(1, dtype=torch.long, device=device)
        # Replace 0 by 1, and 1 by 0
        new_sample.x[:,3][new_sample.x[:,3] == 0] = torch.tensor(2) # 0 -> 2
        new_sample.x[:,3][new_sample.x[:,3] == 1] = torch.tensor(0) # 1 -> 0
        new_sample.x[:,3][new_sample.x[:,3] == 2] = torch.tensor(1) # 2 -> 1
        # new sample as a Dataset object
        dataset.append(new_sample)
        # else:
        #     # We leave the sample but swap the identity of the individuals and activate the first behaviour
        #     dataset[indx2].behaviour[indx_behaviour1] = torch.tensor(1, dtype=torch.long)
        #     dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 0] = torch.tensor(2)
        #     dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 1] = torch.tensor(0)
        #     dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 2] = torch.tensor(1)
   
    return

def rotate_samples(dataset, behaviour, device='cpu'):
    ''' 
    Rotate the samples in the dataset when the behaviour is active.
    '''
    indices = []
    for i in range(len(dataset)):
        if dataset[i].behaviour[behaviour] == torch.tensor(1):
            indices.append(i)
    # Symetric wrt y axis
    for indx in indices:
        new_sample = dataset[indx].clone()
        new_sample.x[:,0] = torch.tensor(1) - new_sample.x[:,0]
        dataset.append(new_sample)
    # Symetric wrt x axis
    for indx in indices:
        new_sample = dataset[indx].clone()
        new_sample.x[:,1] = torch.tensor(1) - new_sample.x[:,1]
        dataset.append(new_sample)
    # Transpose
    for indx in indices:
        new_sample = dataset[indx].clone()
        new_sample.x[:,0], new_sample.x[:,1] = new_sample.x[:,1], new_sample.x[:,0]
        dataset.append(new_sample)
    # Rotate 180 degrees
    for indx in indices:
        new_sample = dataset[indx].clone()
        new_sample.x[:,0] = torch.tensor(1) - new_sample.x[:,0]
        new_sample.x[:,1] = torch.tensor(1) - new_sample.x[:,1]
        dataset.append(new_sample)
    return


    
def downsample_inactive(dataset, idx_behaviour):
    ''' Shuffle before downsampling '''
    indx_inactive = []
    indx_active = []

    for i in range(len(dataset)):
        if dataset[i].behaviour[idx_behaviour] == 1:
            indx_active.append(i)
        elif dataset[i].behaviour[idx_behaviour] == 0:
            indx_inactive.append(i)

    indx_inactive = np.random.choice(indx_inactive, len(indx_active), replace=False)
    indx = np.concatenate((indx_active,  np.random.choice(indx_inactive, len(indx_active), replace=False) ))
    indx = np.random.permutation(indx)
    
    # redefine the dataset
    dataset = [dataset[i] for i in indx]
    return dataset


def downsample_majority_class(dataset, idx_behaviour):
    ''' Downsample the majority class '''
    indx_inactive = []
    indx_active = []

    for i in range(len(dataset)):
        if dataset[i].behaviour[idx_behaviour] == 1:
            indx_active.append(i)
        elif dataset[i].behaviour[idx_behaviour] == 0:
            indx_inactive.append(i)

    if len(indx_active) > len(indx_inactive):
        indx_active = np.random.choice(indx_active, len(indx_inactive), replace=False)
    else:
        indx_inactive = np.random.choice(indx_inactive, len(indx_active), replace=False)

    indx = np.concatenate((indx_active, indx_inactive))
    indx = np.random.permutation(indx)

    # redefine the dataset
    dataset = [dataset[i] for i in indx]
    return dataset
    

