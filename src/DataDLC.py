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
import tqdm
import cv2
import matplotlib.pyplot as plt
#Import Tuple



# Class to handle the data for loading and further processing
class DataDLC:
    ''' Class to handle the data for loading and further processing. '''
    def __init__(self, file = str, detect_jumps = False):
        ''' Constructor of the DataDLC class. It loads the data from the .h5 files and preprocesses it to build the graphs.

            Args:
                file (str): The file to load.'''
        self.file = file
        self.load_data(detect_jumps)

    def load_data(self, detect_jumps):
        ''' Function that loads the data from the .h5 DLC files and preprocesses it to build the graphs.'''
        
        loaded_tab = pd.read_hdf(self.file) # Load the .h5 file

        # Get the scorers
        self.scorer = loaded_tab.columns.levels[0] 

        if len(self.scorer) > 1:
            print('More than one scorer in the .h5 file, the scorers are: ', self.scorer.values)

        #Drop scorer (first level of the columns)
        loaded_tab.columns = loaded_tab.columns.droplevel(0)

        self.individuals =  loaded_tab.columns.levels[0] # Get the individuals

        self.coords_per_indv = []
        for ind in self.individuals: # Save the coordinates per individual to be studied individually
            self.coords_per_indv.append(loaded_tab[ind])

        # Compute the center of mass
        self.compute_center_of_mass()

        # Create a multiindex dataframe for saving the whole configuration in the same dataframe
        # First level: individuals
        # An then as self.coords_per_indv
        self.coords = pd.concat(self.coords_per_indv, axis=1, keys=self.individuals)
        
        self.n_individuals = len(self.individuals) # Get the number of individuals
        self.body_parts = self.coords.columns.levels[1] # Get the body parts
        self.n_body_parts = len(self.body_parts) # Get the number of body parts
        self.n_frames = len(self.coords) # Get the number of frames
        
        #self.cast_boudaries() # Set the boundaries of the individuals 

        self.clean_inconsistent_nans() # Clean the inconsistent NaNs (if any x or y is NaN, set the other to NaN)

        # Save old coordinates
        self.old_coords = self.coords.copy() 


        # Create a mask to indicate where jumps are detected
        self.mask_jumps = pd.DataFrame(index=self.coords.index, columns=self.coords.columns)
        self.mask_jumps = self.mask_jumps.astype(bool)
        self.mask_jumps.loc[:,:] = False
                
        # Eliminate drop y and 'likelihood' columns (only an indicator per body part)
        self.mask_jumps = self.mask_jumps.iloc[:,::3]
        self.mask_jumps = self.mask_jumps.droplevel(2, axis=1)

        if detect_jumps:
            self.detect_isolated_jumps()
            self.remove_outlier_tracklets()
            
        self.fill_nans() # Fill the NaNs with 0

        #self.normalize() # Normalize the coordinates

    def compute_center_of_mass(self):
        # Save the coordinates per individual
        for i, ind in enumerate(self.coords_per_indv):
            x_coords = ind.xs('x', level=1, axis=1)
            y_coords = ind.xs('y', level=1, axis=1)
            likelihood = ind.xs('likelihood', level=1, axis=1)
            # Let's exclude the tail_1, tail_2, tail_3, tail_4 and tail_tip
            x_coords = x_coords.drop(columns=['Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'Tail_tip'])
            y_coords = y_coords.drop(columns=['Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'Tail_tip'])
            likelihood = likelihood.drop(columns=['Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'Tail_tip'])
        
            x_mean = x_coords.mean(axis=1)
            y_mean = y_coords.mean(axis=1)
            likelihood_mean = likelihood.mean(axis=1)

            ind.loc[:, ('Center of mass', 'x')] = x_mean
            ind.loc[:, ('Center of mass', 'y')] = y_mean
            ind.loc[:, ('Center of mass', 'likelihood')] = likelihood_mean
    
    ## NOT WORKING !!
    def cast_boudaries(self):
        ''' Function that sets the boundaries of the coordinates of the individuals. Typically, is [0,640] for x and [0,480] for y. '''

        # Cast the outliers to the boundaries
        for ind in self.individuals:
            for body_part in self.body_parts:
                self.coords[ind].loc[:, (body_part, 'x')].clip(0, 640)
                self.coords[ind].loc[:, (body_part, 'y')].clip(0, 480)
    

    def clean_inconsistent_nans(self):
        ''' If a coordinate x or y is NaN, we set to NaN the other coordinate and the likelihood. '''
        for ind in self.individuals:
            for body_part in self.body_parts:
                x = self.coords[ind].loc[:, (body_part, 'x')]
                y = self.coords[ind].loc[:, (body_part, 'y')]
                frames_to_set_nan = np.where(np.isnan(x) | np.isnan(y))[0]
                self.coords[ind].loc[frames_to_set_nan, (body_part, 'x')] = np.nan
                self.coords[ind].loc[frames_to_set_nan, (body_part, 'y')] = np.nan
                self.coords[ind].loc[frames_to_set_nan, (body_part, 'likelihood')] = np.nan
    
    def fill_nans(self):
        ''' Function that fills the NaNs with 0. '''
        self.coords = self.coords.fillna(0)

    # NOT USED BECAUSE WE CAN'T CAST THE BOUNDARIES (for the moment this is done in dataloader.py, the ideal would be to do it here)
    def normalize(self):
        ''' Function that normalizes the coordinates of the individuals. '''
        for ind in self.individuals:
            # Access the DataFrame for the current individual
            df = self.coords[ind]
            
            # Normalize all 'x' and 'y' values across the DataFrame
            for body_part in self.body_parts:
                # Safely access and normalize x and y coordinates
                if (body_part, 'x') in df.columns:
                    df[(body_part, 'x')] = df[(body_part, 'x')] / 640  # Normalize x
                if (body_part, 'y') in df.columns:
                    df[(body_part, 'y')] = df[(body_part, 'y')] / 480  # Normalize y
            
            # Reassign the normalized DataFrame back to self.coords[ind]
            self.coords[ind] = df

    # NOT USED, MAYBE FOR ANALYSIS OF INDIVIDUAL BEHAVIOUR IS USEFULL 
    # (I don't think it would help in our case, we would lose relative positional information between individuals)
    def center(self):
        ''' Function that centers the individuals wrt the center of mass. '''

        # Center each individual
        for coords_ind in self.coords_per_indv:
            for body_part in self.body_parts:
                coords_ind.loc[:, (body_part, 'x')] -= coords_ind.loc[:, ('Center of mass', 'x')]
                coords_ind.loc[:, (body_part, 'y')] -= coords_ind.loc[:, ('Center of mass', 'y')]

    # SAME AS BEFORE (By normalizing the coordinates individually, we deform the shape of the individuals)
    def min_max_normalization_per_body_part(self):
        ''' Function that performs the Min-Max Normalization for each body part time-serie. '''
        # Min-Max Normalization for each body part
        for ind in self.individuals:
            for body_part in self.body_parts:
                x = self.coords[ind].loc[:, (body_part, 'x')]
                y = self.coords[ind].loc[:, (body_part, 'y')]
                self.coords[ind].loc[:, (body_part, 'x')] = (x - x.min()) / (x.max() - x.min())
                self.coords[ind].loc[:, (body_part, 'y')] = (y - y.min()) / (y.max() - y.min())


    def detect_isolated_jumps(self, threshold_soft_min = 30, threshold_soft_max = 15, imputation =True):
        '''
            Function that detects isolated jumps in the time-series and imputes them with a linear interpolation of the previous and next points.

            Args: 
                threshold_soft_min (int): The soft threshold to detect the jumps. This threshold will detect isolate jumps in the time-series,
                                        where maybe is not as a big jump, but doesn't make sense with the previous and next points.
                threshold_soft_max (int): The soft threshold to detect the jumps. When analysing a jump with the soft threshold, we also check if the consecutive 
                                        points are not more separated than this threshold.
                imputation (bool): If True, the outliers are imputed with the mean of the previous and next points.

            Threshold detection: Let's suppose we have a time-series x = [x_1, x_2, ..., x_n]. At the time t, we detect a jump if:
                - Soft: dist(x_t, x_{t-1}) > threshold_soft_min and dist(x_{t-1}, x_{t+1}) < threshold_soft_max
        '''

         # For each time-series
        for ind in self.individuals:
            for body_part in self.body_parts:
                x = self.coords[ind].loc[:, (body_part, 'x')]
                y = self.coords[ind].loc[:, (body_part, 'y')]
        
                # Compute the euclidiean difference between two consecutive points and two points separated by two frames
                diff = np.stack ([np.sqrt(np.abs(x.diff())**2 + np.abs(y.diff()**2)), np.sqrt(np.abs(x.diff(-1))**2 + np.abs(y.diff(-1))**2), np.sqrt(np.abs(x.diff(2))**2 + np.abs(y.diff(2))**2)], axis=1)

                # If the jump is higher than threshold pixels and the jump of two frames is higher than 50 pixels, set the mask to true
                self.mask_jumps.loc[list(set(np.where(diff[:,0]>threshold_soft_min)[0]).intersection(set(np.where(diff[:,1]>threshold_soft_min)[0])).intersection(set(np.where(diff[:,2]<threshold_soft_max)[0]))), (ind, body_part)] = True

                if imputation:
                    # Set the jumps with interpolation of the previous and next points
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'x')] = ((x.shift() + x.shift(-1))/2).loc[self.mask_jumps.loc[:, (ind, body_part)]] #(2*x - x.diff() - x.diff(-1))/2
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'y')] = ((y.shift() + y.shift(-1))/2).loc[self.mask_jumps.loc[:, (ind, body_part)]] #(2*y - y.diff() - y.diff(-1))/2
                    # Set the likelihood to 0
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'likelihood')] = 0.5
                else:
                    # Impute the jumps with Nans
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'x')] = np.nan
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'y')] = np.nan
                    # Set the likelihood to 0
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'likelihood')] = 0



    def remove_outlier_tracklets(self, threshold_split_tracklets = 30, threshold_jump = 70, percentage_gap_neigh = 0.3, verbose = False):
        ''' Function that removes the outliers tracklets. An outlier tracklet is a tracklet that doesn't concord with the previous and next tracklets.
            
             Args:
                threshold_split_tracklets (int): The threshold to split the tracklets. If the gap between two points is higher than this threshold, a new tracklet is detected.
                threshold_jump (int): The threshold to detect a jump between two tracklets.
                percentage_gap_neigh (float): The percentage of the threshold_jump to be two neighbors tracklets separated by a gap. 

                i.e. A tracklet is removed iff: the jump between the prevoius and itself is higher than the threshold_jump and the gap between the previous and next tracklets is lower than  percentage_gap_neigh*threshold_jump '''

         # For each time-series
        for ind in self.individuals:
            for body_part in self.body_parts:
                x = self.coords[ind].loc[:, (body_part, 'x')]
                y = self.coords[ind].loc[:, (body_part, 'y')]
                # Detect tracklets 
                tracklets = self.detect_tracklets(x, y, threshold=threshold_split_tracklets)
                for t in range(1, len(tracklets)-1):
                    # Check if tracklet is a jump
                    jump_before_x = tracklets[t]['Coords_x'].iloc[0] - tracklets[t-1]['Coords_x'].iloc[-1]
                    jump_before_y = tracklets[t]['Coords_y'].iloc[0] - tracklets[t-1]['Coords_y'].iloc[-1]
                    nans_between_before = tracklets[t]['Frames'].iloc[0] - tracklets[t-1]['Frames'].iloc[-1] - 1
                    gap_between_neigh_tracklets_x = tracklets[t+1]['Coords_x'].iloc[0] - tracklets[t-1]['Coords_x'].iloc[-1]
                    gap_between_neigh_tracklets_y = tracklets[t+1]['Coords_y'].iloc[0] - tracklets[t-1]['Coords_y'].iloc[-1]
                    nans_between_neigh = tracklets[t+1]['Frames'].iloc[0] - tracklets[t]['Frames'].iloc[-1] - 1 + nans_between_before
                    if nans_between_before > 0:
                        jump_before = np.sqrt(jump_before_x**2 + jump_before_y**2)/nans_between_before
                    else:
                        jump_before = np.sqrt(jump_before_x**2 + jump_before_y**2)
                    if nans_between_neigh > 0:
                        gap_neigh_tracklets = np.sqrt(gap_between_neigh_tracklets_x**2 + gap_between_neigh_tracklets_y**2)/nans_between_neigh
                    else:
                        gap_neigh_tracklets = np.sqrt(gap_between_neigh_tracklets_x**2 + gap_between_neigh_tracklets_y**2)
                    
                    if jump_before > threshold_jump and gap_neigh_tracklets < threshold_jump * percentage_gap_neigh:
                        # Set to NaN the tracklet
                        self.coords[ind].loc[tracklets[t]['Frames'], (body_part, 'x')] = np.nan
                        self.coords[ind].loc[tracklets[t]['Frames'], (body_part, 'y')] = np.nan
                        self.coords[ind].loc[tracklets[t]['Frames'], (body_part, 'likelihood')] = np.nan
                        self.mask_jumps[ind].loc[tracklets[t]['Frames'], body_part] = True

                        if verbose:
                            print('Outlier tracklet detected between', tracklets[t]['Frames'].iloc[0], 'and', tracklets[t]['Frames'].iloc[-1], ' of individual', ind, 'and body part', body_part)
                            print('\t Jump between tracklet is', jump_before)

    
    def detect_tracklets(self, x, y, threshold = 30) -> list:
        '''
            Function that detects the tracklets in the time-series. A tracklet is a sequence of points that contains No NaN values. Nan's tracklets are also returned. 
            If the gap between two points is higher than the threshold, a new tracklet is detected.

            Args:
                x (pd.Series): The x time-series.
                y (pd.Series): The y time-series.
                threshold (int): The threshold to split the tracklets. If the gap between two points is higher than this threshold, a new tracklet is detected.

            Returns:
                tracklets (list): The list of tracklets. Each tracklet is a DataFrame with the columns: 'Frames', 'Coords_x', 'Coords_y'.
        '''

        # Get the indices of the NaN values
        nan_indices = np.where(x.isnull())[0]
        # Add the beginning and end of the time-series
        # If there are no NaN values, return the time-series
        if len(nan_indices) == 0:
            nan_indices = np.array([0, len(x)])
        else:
            # Check if beginning and end of the time-series are NaN
            if nan_indices[0] != 0:
                nan_indices = np.concatenate(([0], nan_indices))
            if nan_indices[-1] != len(x):
                nan_indices = np.concatenate((nan_indices, [len(x)]))
            # Get the indices of the tracklets
        tracklets = []
        for i in range(len(nan_indices) - 1):
            if nan_indices[i+1] - nan_indices[i] == 1:
                continue
            else:
                continue_segment_x = x[nan_indices[i]+1:nan_indices[i+1]]
                continue_segment_y = y[nan_indices[i]+1:nan_indices[i+1]]
                gaps_x = continue_segment_x.diff().abs()
                gaps_y = continue_segment_y.diff().abs()
                # If the gap is higher than the threshold, we split the tracklet
                idx_to_split = np.where(np.sqrt(gaps_x**2 + gaps_y**2) > threshold)[0]
                idx_to_split = np.concatenate((idx_to_split, [len(continue_segment_x)]))
                idx_0 = 0
                for idx in idx_to_split:
                    tracklet_x = continue_segment_x[idx_0:idx]
                    tracklet_y = continue_segment_y[idx_0:idx]
                    tracklet_idx = np.arange(nan_indices[i] + 1 + idx_0, nan_indices[i] + 1 + idx)
                    tracklets.append(pd.DataFrame(zip(tracklet_idx, tracklet_x, tracklet_y), columns = ['Frames', 'Coords_x', 'Coords_y']))
                    idx_0 = idx
                
        return tracklets          



    
    def entropy_of_masks(self, mask1, mask2) -> float:
        ''' Function that computes the entropy between two masks.
                
                Args:
                    mask1 (pd.DataFrame): The first mask.
                    mask2 (pd.DataFrame): The second mask.
                
                Returns:
                    entropy (float): The entropy between the two masks. i.e.
                        entropy = - sum_i p_i log(p_i), where p_i is the probability of the i-th element of the mask. '''
        
        return np.sum(np.sum(mask1 != mask2)) / (mask1.values.shape[0] * mask1.values.shape[1])


    def drop_tail_bodyparts(self):
        ''' Function that drops the tail body parts. This function is called before building the graph.
            The tail body parts are: Tail_1, Tail_2, Tail_3, Tail_4 and Tail_tip. '''
    
        self.coords = self.coords.drop(['Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'Tail_tip'], level=1, axis=1)
        self.body_parts = self.body_parts.drop(['Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'Tail_tip'])
        self.n_body_parts = len(self.body_parts)

        


    def create_video(self, video_path, output_path, plot_prev_coords = False, frames = None):
        ''' Function that creates a video with the body parts of the individuals. 
                
                Args:
                    video_path (str): The path to the video.
                    output_path (str): The path to save the video.
                    plot_prev_coords (bool): If True, the previous coordinates of the body parts will be plotted.
                    frames (Tuple): The range of frames to plot. If None, all the frames will be plotted. '''
        
        # Open the video
        cap = cv2.VideoCapture(video_path)

        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        if frames is None:
            frames = range(self.n_frames)
        else:
            frames = range(frames[0], frames[1])

        # set the frame position to the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        # Iterate over the frames
        for i in tqdm.tqdm(frames):#tqdm.tqdm(range(self.n_frames)):
            # Read the frame
            ret, frame = cap.read()

            # Get the coordinates of the individuals
            for j, ind in enumerate(self.individuals):
                for body_part in self.body_parts:
                    x = self.coords[ind].loc[i, (body_part, 'x')]
                    y = self.coords[ind].loc[i, (body_part, 'y')]

                    if plot_prev_coords:
                        x_old = self.old_coords[ind].loc[i, (body_part, 'x')]
                        y_old = self.old_coords[ind].loc[i, (body_part, 'y')]
                        # If Nan, skip                    
                        if not (np.isnan(x_old) or np.isnan(y_old)):
                            # Draw the body parts
                            cv2.circle(frame, (int(x_old), int(y_old)), 6, (120, 120, j*120), -1)
                    if not (np.isnan(x) or np.isnan(y)):
                        # Draw the body parts
                        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, j*255), -1)
                # Add legend to the frame
                cv2.putText(frame, ind, (50, 50 + 50*j), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, j*255), 2, cv2.LINE_AA)

            
            # Write the frame
            out.write(frame)

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def get_statistics_on_jumps(self, plot = False) -> np.ndarray:
        ''' This function will give the mean and standard deviation of the jumps between points for adjency frames. This assumes Gaussian distribution jumps.

            Args:
                plot (bool): If True, the histogram of the jumps will be plotted.
            
            Returns:
                diff (np.ndarray): The jumps between points for adjency frames. '''
        
        self.statistics = {}
        
        for ind in self.individuals:
            for body_part in self.body_parts:
                x = self.coords[ind].loc[:, (body_part, 'x')]
                y = self.coords[ind].loc[:, (body_part, 'y')]
                # Compute the euclidiean difference between two consecutive points and two points separated by two frames
                diff = np.sqrt(np.abs(x.diff())**2 + np.abs(y.diff()**2)) #np.stack ([np.sqrt(np.abs(x.diff())**2 + np.abs(y.diff()**2)), np.sqrt(np.abs(x.diff(-1))**2 + np.abs(y.diff(-1))**2)], axis=1)
                # Get the mean and standard deviation
                mean = diff.mean()
                std = diff.std()
                if plot:
                    plt.hist(diff, bins=100)
                    plt.xlabel('Jump')
                    plt.ylabel('Frequency')
                    plt.title(f'Jump distribution for {ind} and {body_part}')
                    plt.show()
                print(f'Mean of the jumps for {ind} and {body_part}: {mean}')
                print(f'Standard deviation of the jumps for {ind} and {body_part}: {std}')
                self.statistics[(ind, body_part)] = (mean, std)
                break
            break
            
        return diff

    def create_video_per_event(self, video_path, output_path, events, split_behaviour = False):
        ''' Function that creates a video with the tagged events on each frame. If split_behaviour is True, the video will be splitted by the events.

            Args:
                video_path (str): The path to the video.
                output_path (str): The path to save the video.
                events (pd.DataFrame): The events to plot.
                split_behaviour (bool): If True, the video will be splitted by the events.
        '''
        
        # Open the video
        cap = cv2.VideoCapture(video_path)

        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

       

        # Define the individuals
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # set the frame position to the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Iterate over the frames

        # ckeck if the frames of the df are the same as the video
        if len(events) != self.n_frames:
            print('The number of frames in the events dataframe is different than the video')
            return
        
        event_names = events.columns.tolist()
        if split_behaviour:
            for event_name in event_names[1:]:
                # set the frame position to the first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                if events[event_name].sum() == 0:
                    print(f'The event {event_name} is not present in the video')
                    continue

                output_video = os.path.join(output_path, video_path.split('\\')[-1].split('.')[0] + f'_{event_name}.avi')
                out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
                for i in tqdm.tqdm(range(self.n_frames)):
                    # Read the frame
                    ret, frame = cap.read()
                    # Get the event
                    event = events.loc[i, event_name]
                    if event == 1:
                        # write the event in the frame
                        out.write(frame)
                out.release()
        else:
            output_path = os.path.join(output_path, video_path.split('/')[-1].split('.')[0] + f'_events.avi')
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
            for i in tqdm.tqdm(range(self.n_frames)):
                # Read the frame
                ret, frame = cap.read()
                # Get the event
                events_in_frame = event_names[np.where(events.loc[i].values)[0]]
                for e, event in enumerate(events_in_frame):
                    # write the event in the frame
                    cv2.putText(frame, event, (50 + 2*e, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                out.write(frame)
            out.release()


    def save(self, path):
        ''' Function that saves the data to a .h5 file.

            Args:
                path (str): The path to save the file.'''
        # Add scorer on first level of the columns
        a = pd.concat({self.scorer[0]: self.coords.T}, names=['scorer'])
        a.T.to_hdf(path, key='df', mode='w')
