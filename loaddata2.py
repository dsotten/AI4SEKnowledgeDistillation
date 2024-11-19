import json
import re
import pandas as pd
from pathlib import Path
from pprint import pprint
from sklearn.model_selection import train_test_split

# Set up pandas display options for large column content
pd.set_option('max_colwidth', 300)

# Define the function to load and process data
def load_and_process_data(file_paths, output_name):
    """
    Load JSONL files from a list of paths and save selected columns to a CSV file.
    
    file_paths: List of file paths to the JSONL files.
    output_name: The name of the output CSV file.
    """
    columns_long_list = ['repo', 'path', 'func_name', 'original_string',
                         'language', 'code', 'code_tokens', 'docstring',
                         'docstring_tokens', 'sha', 'url']

    def jsonl_list_to_dataframe(file_list, columns=columns_long_list):
        """Load a list of jsonl.gz files into a pandas DataFrame."""
        return pd.concat([pd.read_json(f,
                                       orient='records',
                                       compression='gzip',
                                       lines=True)[columns]
                          for f in file_list], sort=False)

    files = sorted(file_paths)
    print(f'Total number of files to load: {len(files):,}')

    # Load the data into a DataFrame
    df = jsonl_list_to_dataframe(files)
    

    # Select the required columns and save to CSV
    data = df[['code', 'docstring']]
    data.to_csv(output_name, index=False)

# Define the file paths and corresponding output file names for train, test, and valid datasets
trainset_paths = [f'python/final/jsonl/train/python_train_{i}.jsonl' for i in range(13)]
trainset_filename = 'train.csv'

testset_paths = [f'python/final/jsonl/test/python_test_{i}.jsonl' for i in range(13)]
testset_filename = 'test.csv'

evalset_paths = [f'python/final/jsonl/valid/python_valid_{i}.jsonl' for i in range(13)]
evalset_filename = 'eval.csv'

# Load and process train, test, and eval datasets
load_and_process_data(trainset_paths, trainset_filename)
load_and_process_data(testset_paths, testset_filename)
load_and_process_data(evalset_paths, evalset_filename)
