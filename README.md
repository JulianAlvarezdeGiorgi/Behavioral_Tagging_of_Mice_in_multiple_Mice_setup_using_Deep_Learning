# Behavioral_Tagging_of_Mice_in_multiple_Mice_dataset_using_Deep_Learning
*Identification and temporal classification of social behavior of different mice with different pathologies in order to study them.*

The study of social behavior in mice is a crucial aspect of neuroscience research, particularly in understanding the effects of various pathologies or treatments. One of the key methods used in this research is the open-field test, where the behavior and social interactions of mice are observed in an open, stimulus-free environment. Traditionally, analyzing these tests is a manual, time-consuming process, with increasing complexity as the number of mice and interactions grows. Additionally, subjective interpretations can introduce bias, leading to inconsistencies in the results.

This repository addresses these challenges by providing an automated solution for analyzing open-field test videos. Using a top-view configuration, the system leverages DeepLabCut to perform pose estimation, accurately tracking various body parts of the mice. The extracted data is then processed with machine learning tools to identify and quantify specific behaviors. This approach not only streamlines the analysis, reducing the time and effort required, but also improves accuracy and consistency, minimizing human bias in behavior interpretation.

Content:
```
src
├── .ipynb_checkpoints/
├── Analyse_one_video.ipynb
├── analyze.ipynb
├── augmentation.py
├── baseline_models/
├── baseline_models.ipynb
├── Check_graph.ipynb
├── CompressCSV.ipynb
├── DataDLC.py
├── dataloader.py
├── deletefolder/
├── gui.py
├── model.pkl
├── models.py
├── model_sniff_R.pkl
├── preprocessing.py
├── preprocessingTimeSeries.ipynb
├── preprocessingTimeSeries_Oficina.ipynb
├── results_baseline_models.ipynb
├── runs/
├── test.ipynb
├── test_oficina.ipynb
├── train.ipynb
├── train.py
├── train_oficina.ipynb
├── train_oficina_2.ipynb
├── train_poursuit.py
├── utils.py
├── utils_deepof.py
├── Visualization.ipynb
├── __pycache__/
```

- src/dataloader.py: Contains the class Data_DLC, which loads the output file '.h5' of DeepLabCut into a pandas MultiIndex data frame and allows manipulation and pre-processing ot the time-series. Documentation on the specifically functionalities can be found in ...
- src/Models.py: Contains the class that defines the models to be used.
- 
## Description of Files

### `DataDLC.py`
`DataDLC.py` is a core module for loading, preprocessing, and cleaning data from pose estimation .h5 files. This class is designed to handle the raw coordinates of tracked body parts and prepares the data for further analysis, including building graphs for pose-based analysis. Here’s an overview of the class and its main methods:

Class: DataDLC
Handles loading and preprocessing data from .h5 files generated from pose estimation models like DeepLabCut.
Stores key attributes such as the number of individuals, body parts, and frames.
Processes and cleans coordinates to ensure data consistency and prepares it for downstream tasks.
Main Methods:

-**`__init__(self, file: str, detect_jumps: bool = False)`**: Initializes the DataDLC class by loading data from the specified .h5 file and optionally detecting and correcting isolated jumps.

-**`load_data(self, detect_jumps: bool)`**: Loads data from the provided file, extracts coordinates for each individual, computes the center of mass, and handles NaN values and isolated jumps.

-**`compute_center_of_mass(self)`**: Calculates the center of mass for each individual, excluding tail points to focus on core body parts.

-**`clean_inconsistent_nans(self)`**: Ensures that if either the x or y coordinate of a body part is NaN, the entire coordinate set for that frame is set to NaN.

-**`fill_nans(self)`**: Fills NaN values with zeros for consistency in further analysis.

-**`detect_isolated_jumps(self, threshold_soft_min: int, threshold_soft_max: int, imputation: bool)`**: Detects isolated jumps in time-series data using thresholds for minimal and maximal jumps, and optionally imputes these jumps using linear interpolation.

-**`remove_outlier_bouts(self, threshold_split_tracklets: int, threshold_jump: int, percentage_gap_neigh: float, verbose: bool)`**: Identifies and removes outlier tracklets that deviate significantly from neighboring tracklets.

-**`detect_tracklets(self, x, y, threshold: int)`**: Detects tracklets (continuous segments of valid data points) in the time-series data based on specified thresholds.

This class provides a robust way to prepare time-series pose estimation data for further analysis, addressing common issues such as jumps, outliers, and inconsistent coordinates.

### `augmentation.py`

This script contains functions for augmenting and balancing datasets of mouse behavior data, specifically targeting symmetrical behaviors and class imbalances. The functions included are:

- **`merge_symetric_behaviours()`**: Merges two symmetrical behaviors by swapping the identities of the subjects and combining occurrences of the two behaviors into one. This is useful for combining behaviors like 'Sniffing_Resident' and 'Sniffing_Visitor' into a single category while maintaining identity distinctions in the dataset.

- **`rotate_samples()`**: Rotates samples in the dataset based on active behaviors. The function creates symmetry by flipping the pose data along the x or y axis, transposing the coordinates, or rotating them by 180 degrees. This augmentation helps the model generalize better by providing additional variations of the behaviors.

- **`downsample_inactive()`**: Balances the dataset by randomly selecting a subset of inactive samples to match the number of active samples for a specific behavior. This helps in reducing class imbalance, especially when the inactive instances are significantly higher in number.

- **`downsample_majority_class()`**: Downsamples the majority class (either active or inactive samples) to match the count of the minority class for a specified behavior. It aims to maintain class balance, reducing the risk of the model being biased towards the more frequent class.

- **`merge_symetric_behaviours_version2()`**: Similar to `merge_symetric_behaviours()`, but it creates new samples for all instances of a behavior in the secondary individual, preserving additional context. This function is designed to help the model differentiate between individuals while keeping both behaviors represented.

- **`merge_symetric_behaviours_sequences()`**: Applies the merging of symmetrical behaviors on sequences of data, adjusting the identity labels across multiple frames. This is useful for scenarios where the dataset contains time-series data, allowing consistent merging of behaviors across frames while maintaining individual identities.

These functions support data augmentation, balancing, and preparation for training machine learning models on behavior recognition tasks in mice.






