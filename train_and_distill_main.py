import loaddata
import json
import pandas as pd
import json
from pathlib import Path
import ast

# from transformers import RobertaModel
from datasets import Dataset

from setfit import SetFitModel, TrainingArguments, Trainer
from setfit import DistillationTrainer

trainset_path = 'python/final/jsonl/train/python_train.jsonl' #Might need to account for dataset being spread across 13 files
trainset_filename = 'train.csv'
testset_path = 'python/final/jsonl/test/python_test_0.jsonl'
testset_filename = 'test.csv'
evalset_path = 'python/final/jsonl/valid/python_valid_0.jsonl'
evalset_filename = 'eval.csv'

# def merge_json_files(file_paths, output_file):
#     # merged_data = []
#     with open(output_file, "w") as out:
#         for path in file_paths:
#             with open(path, 'r') as file:
#                 for line in file:
#                     out.write(str(json.loads(line))+"\n")
#     # with open(output_file, 'w') as outfile:
#     #     json.dump(merged_data, outfile)

def combine_jsonl_files(input_files, output_file):
    """
    Combine multiple JSONL files into a single JSONL file.
    
    Args:
        input_files (list): List of paths to input JSONL files
        output_file (str): Path to the output JSONL file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Process each input file
        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as infile:
                    # Read line by line to handle large files efficiently
                    for line in infile:
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        # Verify each line is valid JSON
                        try:
                            # Parse and re-serialize to ensure consistent formatting
                            json_obj = json.loads(line.strip())
                            # Write with explicit newline to ensure consistent line endings
                            outfile.write(json.dumps(json_obj) + '\n')
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON in {input_file}: {e}")
            except Exception as e:
                print(f"Error processing {input_file}: {e}")

device = ''
model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", device_map = 'auto')
# model = RobertaModel.from_pretrained("microsoft/codebert-base", device_map = 'auto')

if Path(testset_filename).is_file(): pass
else: loaddata.main(testset_path, testset_filename)

if Path(evalset_filename).is_file(): pass
else: loaddata.main(evalset_path, evalset_filename)

if Path(trainset_filename).is_file(): pass
else: 
    if Path(trainset_path).is_file(): pass
    else:
        trainset_files = []
        for x in range(0,13):
            trainset_files.append('python/final/jsonl/train/python_train_'+str(x)+'.jsonl')
        # merge_json_files(trainset_files,trainset_path)
        combine_jsonl_files(trainset_files,trainset_path)
    loaddata.main(trainset_path, trainset_filename)

train_df = pd.read_csv(trainset_filename)
# train_df['code_tokens'] = train_df['code_tokens'].apply(ast.literal_eval)
# train_df['docstring_tokens'] = train_df['docstring_tokens'].apply(ast.literal_eval)
train_dataset = Dataset.from_pandas(train_df)

test_df = pd.read_csv(testset_filename)
# test_df['code_tokens'] = test_df['code_tokens'].apply(ast.literal_eval)
# test_df['docstring_tokens'] = test_df['docstring_tokens'].apply(ast.literal_eval)
test_dataset = Dataset.from_pandas(test_df)

eval_df = pd.read_csv(evalset_filename)
# eval_df['code_tokens'] = eval_df['code_tokens'].apply(ast.literal_eval)
# eval_df['docstring_tokens'] = eval_df['docstring_tokens'].apply(ast.literal_eval)
unlabeled_train_dataset = Dataset.from_pandas(eval_df)
unlabeled_train_dataset = unlabeled_train_dataset.remove_columns("docstring")
print("Finished loading the datasets")

args = TrainingArguments(
    batch_size=64,
    num_epochs=5,
)

print("Beginning training...")
teacher_trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # column_mapping={"code_tokens": "text", "docstring_tokens": "label"}
    column_mapping={"code": "text", "docstring": "label"}
)
teacher_model = teacher_trainer.train()
metrics = teacher_trainer.evaluate()
print(metrics)
teacher_trainer.save_model('./teacher-model')

model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2", device_map = 'auto')

distillation_args = TrainingArguments(
    batch_size=16,
    max_steps=500,
)

distillation_trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=model,
    args=distillation_args,
    train_dataset=unlabeled_train_dataset,
    eval_dataset=test_dataset,
)

# Train student with knowledge distillation
distilled_model = distillation_trainer.train()
distillation_metrics = distillation_trainer.evaluate()
print(distillation_metrics)
distillation_trainer.save_model('./distilled-model')