import json
import re

import pandas as pd
from pathlib import Path
pd.set_option('max_colwidth',300)
from pprint import pprint

from sklearn.model_selection import train_test_split
# from transformers import RobertaTokenizerFast

def main(file_path, output_name):
    with open(file_path, 'r') as f:
        sample_file = f.readlines()
    sample_file[0]

    pprint(json.loads(sample_file[0]))

    python_files = sorted(Path('python').glob('**/*.gz'))

    print(f'Total number of files: {len(python_files):,}')

    columns_long_list = ['id', 'repository_name', 'func_path_in_repository', 'func_name', 
                        'whole_func_string', 'language', 'func_code_string', 
                        'func_code_tokens', 'func_documentation_string', 'func_documentation_string_tokens',
                        'split_name','func_code_url']

    # columns_short_list = ['code_tokens', 'docstring_tokens', 
    #                     'language', 'partition']

    def jsonl_list_to_dataframe(file_list, columns=columns_long_list):
        """Load a list of jsonl.gz files into a pandas DataFrame."""
        return pd.concat([pd.read_json(f, 
                                    orient='records', 
                                    compression='gzip',
                                    lines=True)[columns] 
                        for f in file_list], sort=False)

    df = jsonl_list_to_dataframe(python_files)

    df.head(3)
    data = df['code']
    data.to_csv(output_name, index=False)