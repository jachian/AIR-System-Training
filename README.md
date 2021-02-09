# AIR-System-Training
Data pre-processing and training of RNN based AIR system

The Training Pipeline
The process of training each classification RNN proceeded as follows:
1.	Step 1 - The file lists of the training data were loaded into memory as their own pandas dataframes. Each dataframe listing the filenames of the CSVs that contain the data flow sequences for the respective set (training, validation and testing).

2.	Step 2 – For each filename listed in a dataframe, the corresponding  y value/output list is generated to represent the binary classification of each flow listed.

3.	Step 3 - Each flow file represented in dataframe is loaded into memory as a list/array of flow steps/sequences.

4.	Step 4 - Sequence Padding/Truncation – Loaded flows that were longer than 10000 steps/sequences are truncated by removing steps 10001 and above. Flows that are shorter than 10000 steps/sequences are pre-padded  with ([0,0,0,0,0,0,0,0,0,0,0,0,0]) entries to make up the LSTM requirement  of 10000 steps per flow. Studies on the effectiveness of pre-padding versus post-padding  for LSTM sequences   suggest that pre-padding is preferable as LSTM models tend to bias towards models towards the post-padding entries the longer the sequences of padding used.

At the end of steps 1 through 4 we would have generated a list of flows sequences (x values) and a corresponding list of classifications (y values) for each the training set. Each flow sequence comprising of 10000 steps with 13 features per step. 

5.	Step 5 - Steps 1 through 4 above are done for the corresponding validation and test lists.

6.	Step 6 - Once the training and validation sets are loaded into memory they were fed into the respective RNN model for training. Learning curves were generated for each training session were used to determine whether the model was underfit, overfit or well fit for the classification task.

7.	Step 7 - For underfit and overfit models the hyper parameters were adjusted and Step 6 repeated until a decent fitting model was trained. The best performing model for each classifier were saved for use with our proposed algorithm.
