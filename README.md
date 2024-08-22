# Behavioral_Tagging_of_Mice_in_multiple_Mice_dataset_using_Deep_Learning
*Identification and temporal classification of social behavior of different mice with different pathologies in order to study them.*

The study of social behavior in mice is a crucial aspect of neuroscience research, particularly in understanding the effects of various pathologies or treatments. One of the key methods used in this research is the open-field test, where the behavior and social interactions of mice are observed in an open, stimulus-free environment. Traditionally, analyzing these tests is a manual, time-consuming process, with increasing complexity as the number of mice and interactions grows. Additionally, subjective interpretations can introduce bias, leading to inconsistencies in the results.

This repository addresses these challenges by providing an automated solution for analyzing open-field test videos. Using a top-view configuration, the system leverages DeepLabCut to perform pose estimation, accurately tracking various body parts of the mice. The extracted data is then processed with machine learning tools to identify and quantify specific behaviors. This approach not only streamlines the analysis, reducing the time and effort required, but also improves accuracy and consistency, minimizing human bias in behavior interpretation.

Content:
- src/dataloader.py: Contains the class Data_DLC, which loads the output file '.h5' of DeepLabCut into a pandas MultiIndex data frame and allows manipulation and pre-processing ot the time-series. Documentation on the specifically functionalities can be found in ...
- src/Models.py: Contains the class that defines the models to be used.
