import os
import cv2
import numpy as np
import pandas as pd

def from_time_to_frame(time, fps):
    time = time.split(':')
    time = time[:-1] + time[2].split('.')
    return int(int(time[0])*3600*fps + int(time[1])*60*fps + int(time[2])*fps + (int(time[3])/1000)*fps)

def from_time_to_frame_s(time, fps):
    return int(time*fps)

def get_video_info(path):
    ''' Return the number of frames and the frame rate of the video in the path'''

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count, fps

def compress_csv(experiment, sex, test, path_to_csvs):
    ''' This creates a single csv file for each video/test with all the behaviour data
        Instead of having the time column, each row is a frame, and each column is a behaviour feature.
        
        experiment: str, name of the experiment: 'DMD', 'MDX5CV', ...
        sex: str, 'male', 'femalle' or 'anesthetized'
        test: str, 'Test 1', 'Test 2', ...
        path_to_csvs: str, path to the folder where the csv files are stored
    '''

    # Get the video/test length, number of frames, frame rate, etc.
    vid_path = os.path.join(path_to_csvs[:-9], 'videos')

    # Experiment path
    if experiment == 'DMD':
        #path = path_to_csvs + 'DMD NULL BEHAVIOUR/'
        path = os.path.join(path_to_csvs, 'DMD NULL BEHAVIOUR')
        name_file = 'SIT_DMD_null_'
        vid_path = os.path.join(vid_path, 'DMD null_mp4')
        name_vid = 'DMD_'

    elif experiment == 'MDX5CV':
        path = path_to_csvs + 'MDX5CV BEHAVIOUR/'
    
    # sex path 
    if sex == 'male':
        #path = path + 'male male/'
        path = os.path.join(path, 'male male')
        name_file = name_file + 'male_male_'
        vid_path = os.path.join(vid_path, vid_path[-12:-4] + ' male videos')
        name_vid = name_vid + 'mal_'
    elif sex == 'femalle':
        path = path + 'male femalle/'

    elif sex == 'anesthetized':
        path = path + 'anesthetized femalle/'

    # Split the two words in the test name, separated by '_'
    test_vid = test.split('_')
    
    video_path = os.path.join(vid_path, name_vid + test_vid[0] + ' ' + test_vid[1]+ '.mp4')
    print(video_path)
    # Create dataframe with each row as a frame
    num_frames, fps = get_video_info(video_path)
    
    # Dataframe to store the compressed data wit a row per frame
    compress_df = pd.DataFrame()
    compress_df['Frame'] = np.arange(num_frames, dtype=int)

    
    # Different files
    folders = ['GENERAL', 'RESIDENT', 'VISITEUR']
    for folder in folders:
        df_list = []
        if folder == 'GENERAL':
            # Get the list of files
            name = name_file + folder
            f_path = os.path.join(path, folder, name, 'TestDataReport_' + name_file + test + '.csv')
            df_list.append(pd.read_csv(f_path))
        elif folder == 'RESIDENT':
            # Only directories
            residents = os.listdir(os.path.join(path, folder))
            # Keep only directories
            residents = [r for r in residents if os.path.isdir(os.path.join(path, folder, r))]
            
            # TO DO: EDIT IT WHEN ALL ANALYSIS ARE DONE
            resident = residents[0]
            #for resident in residents:
            f_path = os.path.join(path, folder, resident, resident + '_t' + test[1:] + '.csv')
            df_list.append(pd.read_csv(f_path))

        elif folder == 'VISITEUR':
            # Only directories
            visitors = os.listdir(os.path.join(path, folder))
            
            # Keep only directories
            visitors = [v for v in visitors if os.path.isdir(os.path.join(path, folder, v))]
            for visitor in visitors:
                f_path = os.path.join(path, folder, visitor, visitor + '_' + test + '.csv')
                df_list.append(pd.read_csv(f_path))

    
        if folder == 'GENERAL':
            # Only the first 2 columns are useful, the first one is the time, we don't need it.

            df_list[0] = df_list[0].iloc[:, :2]

          ## TO DO MAYBE  
        ##elif folder == 'RESIDENT':
            # We have 2 different dataframes
            # Keep all columns
          


        for df in df_list:
            # We need to see if in each frame, each behaviour is present or not
            for behaviour in df.columns[1:]:
                compress_df[behaviour] = np.zeros(num_frames)
                # Get instances of the behaviour
                for i in range(len(df)):

                    starting_frame = from_time_to_frame(df['Time'][i], fps)
                    if i == len(df) - 1:
                        ending_frame = num_frames
                    else:
                        ending_frame = from_time_to_frame(df['Time'][i+1], fps)
                    compress_df[behaviour][starting_frame:ending_frame] = df[behaviour][i]

    return compress_df


def periodogram(signal, method='standard',display = False, window_size=None, overlap=None, window=None):
    """
    Compute and plot the periodogram of a given input signal using the standard, Bartlett, or Welch method.
    
    Parameters:
        signal (array): The input signal.
        method (str): The method to use for periodogram estimation. Options are 'standard' (default), 'bartlett', or 'welch'.
        dispay (bool): If True, plots the Power Spectral Density (dB). 
        window_size (int): The size of the window to use for the Bartlett or Welch methods. If None, defaults to the length of the signal.
        overlap (float): The overlap between segments for the Welch method. Must be a value between 0 and 1. If None, defaults to 0.5.
        window (array): The window to use for the Welch method. The length must be window_size.
    """
    
    n_samples = len(signal)
    if window_size is None:
            window_size = n_samples
    
    if method == 'standard':
        periodogram = np.abs(np.fft.fft(signal))**2 / n_samples
    
    elif method == 'bartlett':
        n_segments = int(np.ceil(n_samples / window_size))
        padded_signal = np.concatenate((signal, np.zeros(window_size * n_segments - n_samples))) # Pad the signal with enough zeros to make its length an integer multiple of window_size
        Per = []
        for i in range(n_segments):
            segment = padded_signal[i*window_size:(i+1)*window_size]
            Per.append(np.abs(np.fft.fft(segment, n = n_samples))**2 / window_size)
        periodogram = np.mean(Per, axis = 0)
    
    elif method == 'welch':
        
        if overlap is None:
            overlap = 0.5
        if window is None:
            window = np.hanning(window_size)
        
        hop_size = int(np.floor(window_size * (1 - overlap)))
        n_segments = int((n_samples - window_size)/hop_size) + 1
        padded_signal = np.concatenate((signal, np.zeros(window_size - n_samples % window_size)))
        segment_indices = np.arange(0, n_samples - window_size + 1, hop_size)
        n_segments = len(segment_indices)
        Per = []
        normalization_P = (1/window_size)*(LA.norm(window)**2)
        for i in range(n_segments):
            segment = padded_signal[segment_indices[i]:segment_indices[i]+window_size]
            Per.append(np.abs(np.fft.fft(segment * window, n = n_samples))**2/(window_size*normalization_P))
        periodogram = np.mean(Per, axis = 0)
    
    else:
        raise ValueError("Invalid method specified. Choose 'standard', 'bartlett', or 'welch'.")
    
    periodogram = np.roll(periodogram, periodogram.size//2)
    
    if display == True:
        f = np.arange(-0.5, 0.5, (1/len(periodogram)))
        plt.plot(f, 20*np.log10(periodogram))
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Power Spectral Density (dB)')
        plt.title(f'Periodogram using {method} method')
        plt.show()

    return periodogram

