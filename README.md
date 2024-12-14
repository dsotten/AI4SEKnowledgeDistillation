This repo stores our progress for developing a distillation process for code summarization generation.

# Training Instructions
To train the model, download loaddata.py and train_and_distill_main.py. Next, obtain the train, test, and eval datasets from the Resources and References section of our final report.
The train dataset has 390k entries, the test dataset has 22k, and the eval data has 23k; use the TRAIN_SIZE, TEST_SIZE, and EVAL_SIZE constants to reduce the size according to the limitations of your machine.
Finally, run train_and_distill_main.py and use the wandb link that pops up to track progress.
Being in the possession of a strong GPU and significant amounts of RAM is recommended to enhance speed of training and prevent catastrophic memory overflows.

# Testing Instructions
When training is complete, there should be two folders: teacher-model and distilled-model.
Simply download and run model_testing.py to output the evaluation of your models.
