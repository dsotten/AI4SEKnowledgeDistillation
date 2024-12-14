This repo stores our progress for developing a distillation process for code summarization generation.
For both training and testing, if you're testing on a different dataset, change trainset_filename, testset_filename, and evalset_filename.
In addition, alter the train_size, test_size, and eval_size variables to use a smaller subset of the data if needed.

# Training Instructions
To train the model, download loaddata.py and train_and_distill_main.py. Next, obtain the train, test, and eval datasets from the Resources and References section of our final report.
If the jsonl files from CodeSearchNet are used, replace trainset_path, testset_path, and evalset_path with your file paths.
Finally, run train_and_distill_main.py and use the wandb link that pops up to track progress.
Being in the possession of a strong GPU and significant amounts of RAM is recommended to enhance speed of training and prevent catastrophic memory overflows.

# Testing Instructions
When training is complete, there should be two folders: teacher-model and distilled-model.
Simply download and run model_testing.py to output the evaluation of your models.

# Requirements
transformers<=4.42.2, pandas, json, ast, torch, setfit, sklearn. Any other dependencies will be installed along the way.
