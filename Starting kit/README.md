# Readme for ComMA@ICON Shared Task Starting Kit

This is a sample starting kit for the Shared Task on Multilingual Gender Bias and Communal Language Identification.

This kit contains the following files / directories -
- A sample submission zip, if you are participating for all the languages including the multilingual dataset. It depicts the naming pattern of the file and the format of the file itself as is expected (pay close attention to the file extension, header inside the file, field separator). If you are participating for one or some of the languages then everything remains same except the number of files - please note even if you have one file to submit, it will still need to be zipped.
- The scoring program being used for the task. It contains three functions 
  - `validate_format` - for validating the submission file (needs two dataframes - one for gold and the other for prediction).
  - `get_microf1` - calculates and returns micro-F1 for the given predictions. It returns the individual micro-F1 scores for each of the three classes as well as an overall score. It takes in a dataframe with at least two columns - 'Labels_x' (gold labels) and 'Labels_y' (predicted labels).
  -  `get_instancef1` - calculates and returns instance-F1 for the given predictions. It takes in a dataframe with at least two columns - 'Labels_x' (gold labels) and 'Labels_y' (predicted labels).
- A random baseline generator. It could be directly run to produce random baselines and predictions. It provides some utility functions for reading the training and test files and also generating the predictions files.
- A copy of the training and development files (these are available for download separately as well and both are exactly same copies).