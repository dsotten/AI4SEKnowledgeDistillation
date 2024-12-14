This repo stores our progress for developing a distillation process for code summarization generation.

# Training Instructions
To train the model, download loaddata.py and train_and_distill_main.py. Next, obtain the train, test, and eval datasets from the Resources and References section of our final report.
If you're training with a different dataset, change train_csv, test_csv, and eval_csv accordingly.
Finally, run train_and_distill_main.py and use the wandb link that pops up to track progress.
Being in the possession of a strong GPU and significant amounts of RAM is recommended to enhance speed of training and prevent catastrophic memory overflows.

# Testing Instructions
When training is complete, there should be two folders: teacher-model and distilled-model.
Simply download and run model_testing.py to output the evaluation of your models.
If you're testing on a different dataset, change train_csv, test_csv, and eval_csv according
