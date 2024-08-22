# DATALOADER CLASS to handle the data loading and preprocessing
# We load the .h5 files with the trajectories of DeepLabCut and preprocess them to build the graps
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import time
from statsmodels.tsa.arima.model import ARIMA

import h5py
import numpy as np
import os
import torch
#import utils as ut
import pandas as pd
import tqdm
import cv2
import matplotlib.pyplot as plt


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
        self.scorer = loaded_tab.columns.levels[0]
        if len(self.scorer) > 1:
            print('More than one scorer in the .h5 file, the scorers are: ', scorer.values)
        #Drop scorer (first level of the columns)
        loaded_tab.columns = loaded_tab.columns.droplevel(0)

        self.individuals =  loaded_tab.columns.levels[0] # Get the individuals
        self.coords_per_indv = []
        for ind in self.individuals: # Save the coordinates per individual to be studied individually
            self.coords_per_indv.append(loaded_tab[ind])
        self.compute_center_of_mass()
        # Create a multiindex dataframe for saving the whole configuration in the same dataframe
        # First level: individuals
        # An then as self.coords_per_indv
        self.coords = pd.concat(self.coords_per_indv, axis=1, keys=self.individuals)
        
        self.n_individuals = len(self.individuals) # Get the number of individuals
        self.body_parts = self.coords.columns.levels[1] # Get the body parts
        self.n_body_parts = len(self.body_parts) # Get the number of body parts
        self.n_frames = len(self.coords) # Get the number of frames

        self.clean_inconsistent_nans() # Clean the inconsistent NaNs

        # Save old coordinates
        self.old_coords = self.coords.copy()

        # Create a mask to indicate where jumps are detected
        self.mask_jumps = pd.DataFrame(index=self.coords.index, columns=self.coords.columns)
        self.mask_jumps = self.mask_jumps.astype(bool)
        self.mask_jumps.loc[:,:] = False
                
        # Eliminate drop y and 'likelihood' columns
        self.mask_jumps = self.mask_jumps.iloc[:,::3]
        self.mask_jumps = self.mask_jumps.droplevel(2, axis=1)

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


    def center(self):
        ''' Function that centers the individuals in the configuration. '''

        # Center each individual
        for coords_ind in self.coords_per_indv:
            for body_part in self.body_parts:
                coords_ind.loc[:, (body_part, 'x')] -= coords_ind.loc[:, ('Center of mass', 'x')]
                coords_ind.loc[:, (body_part, 'y')] -= coords_ind.loc[:, ('Center of mass', 'y')]

    def min_max_normalization_per_body_part(self):
        ''' Function that performs the Min-Max Normalization for each body part time-serie. '''
        # Min-Max Normalization for each body part
        for ind in self.individuals:
            for body_part in self.body_parts:
                x = self.coords[ind].loc[:, (body_part, 'x')]
                y = self.coords[ind].loc[:, (body_part, 'y')]
                self.coords[ind].loc[:, (body_part, 'x')] = (x - x.min()) / (x.max() - x.min())
                self.coords[ind].loc[:, (body_part, 'y')] = (y - y.min()) / (y.max() - y.min())

        # Min-Max Normalization for
    def detect_outliers(self, imputation=False):
        ''' Function that detects the outliers by fitting an ARMA model to each time-series and discarding the points are farther than 3 standard deviations from the prediction.
            
             Args:
                imputation (bool): If True, the outliers are imputed with the prediction of the ARMA model. '''


        # For each time-series
        for ind in self.individuals:
            for body_part in self.body_parts:
                x = self.coords[ind].loc[:, (body_part, 'x')]
                y = self.coords[ind].loc[:, (body_part, 'y')]
                # Fit the ARMA model
                model_x = ARIMA(x, order=(2, 1, 2))
                model_y = ARIMA(y, order=(2, 1, 2))
                model_x_fit = model_x.fit()
                model_y_fit = model_y.fit()
                # Predict the time-series
                x_pred = model_x_fit.predict()
                y_pred = model_y_fit.predict()
                # Compute the residuals
                residuals_x = x - x_pred
                residuals_y = y - y_pred
                # Compute the standard deviation
                std_x = residuals_x.std()
                std_y = residuals_y.std()
                # Detect the outliers
                if imputation:
                    self.coords[ind].loc[residuals_x > 3 * std_x, (body_part, 'x')] = x_pred[residuals_x > 3 * std_x]
                    self.coords[ind].loc[residuals_y > 3 * std_y, (body_part, 'y')] = y_pred[residuals_y > 3 * std_y]
                else:
                    self.coords[ind].loc[residuals_x > 3 * std_x, (body_part, 'x')] = np.nan
                    self.coords[ind].loc[residuals_y > 3 * std_y, (body_part, 'y')] = np.nan

    def detect_isolated_jumps(self, threshold_soft_min = 30, threshold_soft_max = 15, imputation = False):
        '''
            Function that detects isolated jumps in the time-series and imputes them with a linear interpolation of the previous and next points.

            Args: 
                threshold_soft_min (int): The soft threshold to detect the jumps. This threshold will detect isolate jumps in the time-series,
                                        where maybe is not as a big jump, but doesn't make sense with the previous and next points.
                threshold_soft_max (int): The soft threshold to detect the jumps. When analysing a jump with the soft threshold, we also check if the consecutive 
                                        points are not more separated than this threshold.
                imputation (bool): If True, the outliers are imputed with the prediction of the ARMA model.

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
                    # Impute the jumps
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'x')] = self.ARIMA(x, self.mask_jumps.loc[:, (ind, body_part)])
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'y')] = self.ARIMA(y, self.mask_jumps.loc[:, (ind, body_part)])
                    # Impute the likelihood with the mean of the likelihood of the previous and next points
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'likelihood')] = 0
                else:
                    # Set the jumps with interpolation of the previous and next points
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'x')] = ((x.shift() + x.shift(-1))/2).loc[self.mask_jumps.loc[:, (ind, body_part)]] #(2*x - x.diff() - x.diff(-1))/2
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'y')] = ((y.shift() + y.shift(-1))/2).loc[self.mask_jumps.loc[:, (ind, body_part)]] #(2*y - y.diff() - y.diff(-1))/2
                    # Set the likelihood to 0
                    self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'likelihood')] = 0.5


    def detect_jumps(self, threshold_hard = 70, threshold_soft_min = 30, threshold_soft_max = 15, imputation = False):
        ''' Function that detects the jumps in the time-series and imputes them with Nans or the prediction of the ARMA model.

            Args:
                threshold_hard (int): The hard threshold to detect the jumps. This threshold will detect unatural jumps in the time-series.
                threshold_soft_min (int): The soft threshold to detect the jumps. This threshold will detect isolate jumps in the time-series,
                                        where maybe is not as a big jump, but doesn't make sense with the previous and next points.
                threshold_soft_max (int): The soft threshold to detect the jumps. When analysing a jump with the soft threshold, we also check if the consecutive 
                                        points are not more separated than this threshold.
                imputation (bool): If True, the outliers are imputed with the prediction of the ARMA model. 

                Threshold detection: Let's suppose we have a time-series x = [x_1, x_2, ..., x_n]. At the time t, we detect a jump if:
                    - Hard: dist(x_t, x_{t-1}) > threshold_hard
                    - Soft: dist(x_t, x_{t-1}) > threshold_soft_min and dist(x_{t-1}, x_{t+1}) < threshold_soft_max
        '''


                # Now hard jumps are detected, we do it separately to avoid the jumps detected with the soft threshold to influence the hard threshold

                # x = self.coords[ind].loc[:, (body_part, 'x')]
                # y = self.coords[ind].loc[:, (body_part, 'y')]
        
                # # Compute the euclidiean difference between two consecutive points and two points separated by two frames
                # diff = np.stack ([np.sqrt(np.abs(x.diff())**2 + np.abs(y.diff()**2)), np.sqrt(np.abs(x.diff(-1))**2 + np.abs(y.diff(-1))**2), np.sqrt(np.abs(x.diff(2))**2 + np.abs(y.diff(2))**2)], axis=1)

                # # Detect the jumps
                # self.mask_jumps.loc[diff[:,0] > threshold_hard, (ind, body_part)] = True

                # if imputation:
                #     # Impute the jumps
                #     self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'x')] = self.ARIMA(x, self.mask_jumps.loc[:, (ind, body_part)])
                #     self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'y')] = self.ARIMA(y, self.mask_jumps.loc[:, (ind, body_part)])
                #     # Impute the likelihood with the mean of the likelihood of the previous and next points
                #     self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'likelihood')] = 0
                # else:
                #     # Set the jumps with interpolation of the previous and next points
                #     self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'x')] = ((x.shift() + x.shift(-1))/2).loc[self.mask_jumps.loc[:, (ind, body_part)]] #(2*x - x.diff() - x.diff(-1))/2
                #     self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'y')] = ((y.shift() + y.shift(-1))/2).loc[self.mask_jumps.loc[:, (ind, body_part)]] #(2*y - y.diff() - y.diff(-1))/2
                #     # Set the likelihood to 0
                #     self.coords[ind].loc[self.mask_jumps.loc[:, (ind, body_part)], (body_part, 'likelihood')] = 0.5
                
    def iterative_jump_detection(self, threshold_hard = [100, 50], threshold_soft_min = 30, threshold_soft_max = 15, tol = 1e-3, max_iter = 50, imputation = False, verbose = False):
        ''' 
            Iterates over the jump detection and imputation until the mask of jumps doesn't change.
        '''

        # First the isolated jumps are detected
        self.detect_isolated_jumps(threshold_soft_min, threshold_soft_max, imputation=imputation)

        # Iterate over the jump detection and imputation
        for i in tqdm.tqdm(range(max_iter)):
            mask_jumps_prev = self.mask_jumps.copy()
            self.detect_jumps(threshold_hard[0], threshold_soft_min, threshold_soft_max, imputation=False)
            if threshold_hard[0] > threshold_hard[1]:
                threshold_hard[0]*=0.9
            else:
                threshold_hard[0] = threshold_hard[1]
            if verbose:
                print(f'Entropy of the mask: {self.entropy_of_masks(self.mask_jumps, mask_jumps_prev)},  Number of jumps detected: {self.mask_jumps.sum().sum() - mask_jumps_prev.sum().sum()}', end='\r')
            if self.entropy_of_masks(self.mask_jumps, mask_jumps_prev) < tol:
                if imputation:
                    self.impute_with_ARIMA()                        
                break
        if verbose:
            print('\n')
            print(f'Total number of jumps detected: {self.mask_jumps.sum().sum()}')

    def remove_outlier_bouts_old(self, threshold, verbose = False):
        ''' Function that removes the outliers bouts. An outlier bout is a tracklet that doesn't concord with the previous and next tracklets. '''
    

        # For each time-series
        for ind in self.individuals:
            for body_part in self.body_parts:
                x = self.coords[ind].loc[:, (body_part, 'x')]
                y = self.coords[ind].loc[:, (body_part, 'y')]
                # Detect tracklets 
                tracklets_x = self.detect_tracklets(x)
                tracklets_y = self.detect_tracklets(y)

                # For each tracklet, see the jump between the previous and next tracklet
                for t in range(2, len(tracklets_x)):
                    # Check if is a NaN tracklet
                    if tracklets_x[t]['Coords'].isna().any():
                        continue
                    # Get the jump between the previous and next tracklet
                    jump_x = tracklets_x[t].iloc[0, 1] - tracklets_x[t-2].iloc[-1, 1]
                    jump_y = tracklets_y[t].iloc[0, 1] - tracklets_y[t-2].iloc[-1, 1]
                    num_NaN_between = len(tracklets_x[t-1])
                    #print('Jump between tracklet', t-1, 'and', t, 'is', jump_x, jump_y, 'with', num_NaN_between, 'NaNs in between')
                    # If the jump is higher than the threshold, set the tracklet to NaN
                    if (np.sqrt(jump_x**2 + jump_y**2) / num_NaN_between) > threshold:
                        self.coords[ind].loc[tracklets_x[t].iloc[0, 0]:tracklets_x[t].iloc[-1, 0], (body_part, 'x')] = np.nan
                        self.coords[ind].loc[tracklets_y[t].iloc[0, 0]:tracklets_y[t].iloc[-1, 0], (body_part, 'y')] = np.nan
                        self.coords[ind].loc[tracklets_x[t].iloc[0, 0]:tracklets_x[t].iloc[-1, 0], (body_part, 'likelihood')] = 0
                        self.mask_jumps[ind][body_part].loc[tracklets_x[t].iloc[0, 0]:tracklets_x[t].iloc[-1, 0]] = True
                        if verbose:
                            print('Outlier bout detected between', tracklets_x[t-2].iloc[-1, 0], 'and', tracklets_x[t].iloc[0, 0], ' of individual', ind, 'and body part', body_part)
                            print('\t Jump between tracklet is', np.sqrt(jump_x**2 + jump_y**2), 'with', num_NaN_between, 'NaNs in between')

    def remove_outlier_bouts(self, threshold_split_tracklets = 30, threshold_jump = 70, percentage_gap_neigh = 0.3, verbose = False):
        ''' Function that removes the outliers bouts. An outlier bout is a tracklet that doesn't concord with the previous and next tracklets.
            
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
                        self.coords[ind].loc[tracklets[t]['Frames'], (body_part, 'likelihood')] = 0
                        self.mask_jumps[ind].loc[tracklets[t]['Frames'], body_part] = True

                        if verbose:
                            print('Outlier bout detected between', tracklets[t]['Frames'].iloc[0], 'and', tracklets[t]['Frames'].iloc[-1], ' of individual', ind, 'and body part', body_part)
                            print('\t Jump between tracklet is', jump_before)

    def detect_tracklets_old(self, x):
        ''' Function that detects the tracklets in the time-series. A tracklet is a sequence of points that contains No NaN values. Nan's tracklets are also returned. '''

        # Get the indices of the NaN values
        nan_indices = np.where(x.isnull())[0]
        # Get the indices of the tracklets
        tracklets = []
        tracklet_nan = []
        tracklet_indx = []
        for i in range(len(nan_indices) - 1):
            if nan_indices[i+1] - nan_indices[i] == 1:
                tracklet_nan.append(x[nan_indices[i]])
                tracklet_indx.append(nan_indices[i])

            else:
                tracklet_nan.append(np.nan)
                tracklet_indx.append(nan_indices[i])
                tracklets.append(pd.DataFrame(zip(tracklet_indx, tracklet_nan), columns=['Frames', 'Coords']))

                tracklet_indx = np.arange(nan_indices[i] + 1, nan_indices[i+1])
                tracklets.append(pd.DataFrame(zip(tracklet_indx, x[nan_indices[i]+1:nan_indices[i+1]]), columns=['Frames', 'Coords']))

                tracklet_nan = []
                tracklet_indx = []

        return tracklets
    
    def detect_tracklets(self, x, y, threshold = 1):
        '''
            Function that detects the tracklets in the time-series. A tracklet is a sequence of points that contains No NaN values. Nan's tracklets are also returned. 
            If the gap between two points is higher than the threshold, a new tracklet is detected.
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



        
    def entropy_of_masks(self, mask1, mask2):
        ''' Function that computes the entropy between two masks. '''
        return np.sum(np.sum(mask1 != mask2)) / (mask1.values.shape[0] * mask1.values.shape[1])

    def impute_with_ARIMA(self, order = (2, 1, 2)):
        ''' Function that imputes the jumps with the prediction of the ARIMA model. '''

        # For each time-series
        for ind in self.individuals:
            for body_part in self.body_parts:
                x_time_serie = self.coords[ind][body_part]['x'][~self.mask_jumps[ind][body_part]]
                y_time_serie = self.coords[ind][body_part]['y'][~self.mask_jumps[ind][body_part]]
                # Fit the ARIMA model
                model_x = ARIMA(x_time_serie, order=order)
                model_y = ARIMA(y_time_serie, order=order)
                model_x_fit = model_x.fit()
                model_y_fit = model_y.fit()
                # Predict the time-series
                x_pred = model_x_fit.predict()
                y_pred = model_y_fit.predict()
                # Impute the jumps
                self.coords[ind][body_part]['x'][self.mask_jumps[ind][body_part]] = x_pred
        

    def create_video(self, video_path, output_path, plot_prev_coords = False, frames = None):
        ''' Function that creates a video with the body parts of the individuals. '''
        # Open the video
        cap = cv2.VideoCapture(video_path)

        # Get the width and height of the video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        # Define the individuals
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

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
                            cv2.circle(frame, (int(x_old), int(y_old)), 6, (255, 255, j*255), -1)
                    if not (np.isnan(x) or np.isnan(y)):
                        # Draw the body parts
                        cv2.circle(frame, (int(x), int(y)), 4, colors[j], -1)

            # Add legend to the frame
            cv2.putText(frame, 'Individual 1', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Individual 2', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Write the frame
            out.write(frame)

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def get_statistics_on_jumps(self, plot = False):
        ''' This function will give the mean and standard deviation of the jumps between points for adjency frames. This assumes Gaussian distribution jumps.

            Args:
                plot (bool): If True, the histogram of the jumps will be plotted. '''
        
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



            
        



    def ARIMA(self, x, mask):
        ''' Function that fits an ARIMA model to a time-series and predicts the missing values.

            Args:
                x (pd.Series): The time-series.
                mask (pd.Series): The mask indicating the missing values.

            Returns:
                x_pred (pd.Series): The predicted time-series. '''
        
        # Fit the ARIMA model
        model = ARIMA(x[~mask], order=(2, 1, 2))
        model_fit = model.fit()
        # Predict the time-series
        x_pred = model_fit.predict()
        return x_pred

    def preprocess(self):
        ''' Function that preprocesses the data. '''
        #self.center()
        #self.min_max_normalization_per_body_part()
        #self.detect_outliers()


    def save(self, path):
        ''' Function that saves the data to a .h5 file.

            Args:
                path (str): The path to save the file.'''
        # Add scorer on first level of the columns
        a = pd.concat({self.scorer[0]: self.coords.T}, names=['scorer'])
        a.T.to_hdf(path, key='df', mode='w')

class DLCDataLoader(DataLoader):
    ''' The DataLoader class for the DeepLabCut data. It loads the data from the .h5 files and preprocesses it to build the graphs. '''
    
    def __init__(self, root, batch_size = 1, num_workers = 1, device = 'cpu', window_size=None, stride=None, build_graph=False):
        ''' Constructor of the DataLoader class. It loads the data from the .h5 files and preprocesses it to build the graphs.

            Args:
                root (str): The root directory of the .h5 files.
                batch_size (int): The batch size.
                num_workers (int): The number of workers for the DataLoader.
                device (torch.device): The device to load the data.
                window_size (int): The window size for the temporal graph.
                stride (int): The stride for the temporal graph.
                spatio_temporal_adj (MultiIndex): The spatio-temporal adjacency matrix.
                build_graph (bool): If True, the graph is built from the coordinates of the individuals. '''

        self.root = root
        self.batch_size = batch_size # Batch size
        self.num_workers = num_workers # Number of workers for the DataLoader
        self.device = device # Device to load the data
        self.window_size = window_size
        self.stride = stride # Stride for the temporal graph
        self.buid_graph = build_graph
        
        
        self.files = [f for f in os.listdir(root) if f.endswith('filtered.h5')]
        # Order by number of the test
        self.files.sort(key=lambda x: int(x.split('DLC')[0].split('_')[3]))
        #self.files.sort(key=lambda x: int(x.split('DLC')[0].split('_')[2].split(' ')[1]))
        #self.files.sort()
        self.n_files = len(self.files) # Number of files, i.e. number of spatio-temporal graphs
        self.dataset = []
        self.behaviour = []
       
        print(f"Loading data from {root}, where we have {self.n_files} files")
        self.load_data_2()	# Load the data

        print(f"Number of files: {self.n_files}")

        super(DLCDataLoader, self).__init__(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def __len__(self):
        ''' Function that returns the number of files. '''
        return self.n_files
    
    def __getitem__(self, idx):
        ''' Function that returns the data at a given index.

            Args:
                idx (int): The index of the data.

            Returns:
                data (Data): The data at the given index.'''
        return self.dataset[idx]
    
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

        return node_features, edge_index, frame_mask
    

    ####### TO OPTIMISE, IT TAKES TOO LONG, PROBABLY BECAUSE OF THE DATA CLASS MANEGMENT, IT TAKES TO LONG TO ACCESS THE DATA #######
    def build_graph_2(self, data_dlc):
        ''' 
        Function that builds the graph from an instance of DataDLC class.

        Args:
            data_dlc (DataDLC): The instance of the DataDLC class.
            spatio_temporal_adj (MultiIndex): The spatio-temporal adjacency matrix.

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
                    for body_part_2 in data_dlc.body_parts[:body_part_idx]:
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
                for ind_idx_2, individual_2 in enumerate(data_dlc.individuals[:ind_idx]):
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

    def load_data_2(self):
        '''
        Function that loads the data from the .h5 files and preprocesses it to build the graphs.
        It uses the DataDLC class to load the data. 
        '''                
        
        print(f"We have {self.n_files} files")
        for i, file in tqdm.tqdm(enumerate(self.files)):
            # Load the data
            print(f"Loading file {file}")
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
            dlc = DataDLC(os.path.join(self.root, file))
            dlc.preprocess()
            
            if self.buid_graph:
                # Build the graph
                t0 = time.time()
                node_features, edge_index, frame_mask = self.build_graph_2(dlc)
                print(f"Graph built in {time.time() - t0} s")
                data = Data(x=node_features, edge_index=edge_index, y=behaviour, frame_mask=frame_mask, file=file)
                self.dataset.append(data)
            else:
                self.dataset.append((dlc.coords, behaviour))
        
    def load_data_3(self):
        '''
        Function that loads the data from the .h5 files and preprocesses it to build the graphs.
        It uses the DataDLC class to load the data. 
        '''                
        
        csv_files = [f for f in os.listdir(self.root) if f.endswith('.csv')]

        print(f"We have {self.n_files} files")
        for i, file in tqdm.tqdm(enumerate(self.files)):
            # Load the data
            print(f"Loading file {file}")
            # Name of the test
            name_test = file.split('DLC')[0]

            # Load the behaviour
            # See if there's a behaviour file
            num_test = int(name_test.split('_')[2].split(' ')[1])	
            name_file = [f for f in csv_files if f.split('_')[3][:-4] == str(num_test)]
            print(f"Name file: {name_file}")
            if len(name_file) == 0:
                print(f"No behaviour file for {name_test}")
                behaviour = None
            else:
                behaviour = self.load_behaviour(name_file[0])
                #behaviour = torch.tensor(behaviour.values, dtype=torch.int64)


            # Load the data
            dlc = DataDLC(os.path.join(self.root, file))
            dlc.preprocess()
            
            if self.buid_graph:
                # Build the graph
                t0 = time.time()
                node_features, edge_index, frame_mask = self.build_graph_2(dlc)
                print(f"Graph built in {time.time() - t0} s")
                data = Data(x=node_features, edge_index=edge_index, y=behaviour, frame_mask=frame_mask, file=file)
                self.dataset.append(data)
            else:
                self.dataset.append((dlc.coords, behaviour))
        
        
                
            

    def load_data(self):
        i = 0
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
                data = Data(x=node_features, edge_index=edge_index, y=file, frame_mask=frame_mask, behaviour=[behaviour])
                print(f"Graph built in {time.time() - t0} s")
                self.dataset.append(data)
                continue
            
            # Slide the window to build the differents graphs
            for j in tqdm.tqdm(range(0, n_frames - self.window_size + 1, self.stride)):

                # Behaviour in the window
                behaviour_window = behaviour[j:j+self.window_size]

                # Build the graph
                node_features, edge_index, frame_mask = self.build_graph(coords_indv[:, j:j+self.window_size, :])

                # Build the data object
                data = Data(x=node_features, edge_index=edge_index, y=file, frame_mask=frame_mask, behaviour=behaviour_window)
                self.dataset.append(data)
            
    def load_behaviour(self, file):
        ''' Function that loads the behaviour from a csv file.

            Args:
                file (str): The csv file to load.

            Returns:
                behaviour (torch.Tensor): The behaviour as a tensor.'''
        
        return pd.read_csv(os.path.join(self.root, file))


    


        


