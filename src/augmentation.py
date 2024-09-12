import numpy as np
import torch

def merge_symetric_behaviours(indx_behaviour1, indx_behaviour2, dataset):

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
            new_sample.behaviour = np.zeros(len(new_sample.behaviour))
            new_sample.behaviour[indx_behaviour1] = 1
            # Replace 0 by 1, and 1 by 0
            new_sample.x[:,3][new_sample.x[:,3] == 0] = 2
            new_sample.x[:,3][new_sample.x[:,3] == 1] = 0
            new_sample.x[:,3][new_sample.x[:,3] == 2] = 1
            dataset.append(new_sample)
        else:
            # We leave the sample but swap the identity of the individuals and activate the first behaviour
            dataset[indx2].behaviour[indx_behaviour1] = 1
            dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 0] = 2
            dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 1] = 0
            dataset[indx2].x[:,3][dataset[indx2].x[:,3] == 2] = 1
   
    return
   


    

