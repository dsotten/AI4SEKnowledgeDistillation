import loaddata
import pandas as pd
import json
from pathlib import Path
import ast

from transformers import Trainer
from datasets import Dataset

from setfit import SetFitModel, TrainingArguments, Trainer
from setfit import DistillationTrainer

trainset_path = 'python/final/jsonl/train/python_train_0.jsonl' #Might need to account for dataset being spread across 13 files
trainset_filename = 'train.csv'
testset_path = 'python/final/jsonl/train/python_test_0.jsonl'
testset_filename = 'test.csv'
evalset_path = 'python/final/jsonl/train/python_valid_0.jsonl'
evalset_filename = 'eval.csv'

device = ''
model = SetFitModel.from_pretrained("microsoft/codebert-base", device_map = 'auto')

if Path(trainset_filename).is_file(): pass
else: loaddata.main(trainset_path, trainset_filename)

if Path(testset_filename).is_file(): pass
else: loaddata.main(testset_path, testset_filename)

if Path(evalset_filename).is_file(): pass
else: loaddata.main(evalset_path, evalset_filename)

train_df = pd.read_csv(trainset_filename)
train_df['func_code_tokens'] = train_df['func_code_tokens'].apply(ast.literal_eval)
train_df['func_documentation_string_tokens'] = train_df['func_documentation_string_tokens'].apply(ast.literal_eval)
train_dataset = Dataset.from_pandas(train_df)

test_df = pd.read_csv(testset_filename)
test_df['func_code_tokens'] = test_df['func_code_tokens'].apply(ast.literal_eval)
test_df['func_documentation_string_tokens'] = test_df['func_documentation_string_tokens'].apply(ast.literal_eval)
test_dataset = Dataset.from_pandas(test_df)

eval_df = pd.read_csv(evalset_filename)
eval_df['func_code_tokens'] = eval_df['func_code_tokens'].apply(ast.literal_eval)
eval_df['func_documentation_string_tokens'] = eval_df['func_documentation_string_tokens'].apply(ast.literal_eval)
eval_dataset = Dataset.from_pandas(eval_df)


args = TrainingArguments(
    batch_size=64,
    num_epochs=5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
teacher_model = trainer.train()

metrics = trainer.evaluate()
print(metrics)


distillation_args = TrainingArguments(
    batch_size=16,
    max_steps=500,
)

distillation_trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=model,
    args=distillation_args,
    train_dataset=unlabeled_train_dataset,
    eval_dataset=eval_dataset,
)

# Train student with knowledge distillation
distilled_model = distillation_trainer.train()
distillation_metrics = distillation_trainer.evaluate()
print(distillation_metrics)