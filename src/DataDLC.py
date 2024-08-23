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
#Import Tuple



# Class to handle the data for loading and further processing
class DataDLC:
    ''' Class to handle the data for loading and further processing. '''
    def __init__(self, file = str):
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
        print(self.scorer)
        if len(self.scorer) > 1:
            print('More than one scorer in the .h5 file, the scorers are: ', self.scorer.values)
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
        
    def drop_tail_bodyparts(self):
        ''' Function that drops the tail body parts. This function is called before building the graph. '''
    
        self.coords = self.coords.drop(['Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'Tail_tip'], level=1, axis=1)
        self.body_parts = self.body_parts.drop(['Tail_1', 'Tail_2', 'Tail_3', 'Tail_4', 'Tail_tip'])
        self.n_body_parts = len(self.body_parts)

        


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
