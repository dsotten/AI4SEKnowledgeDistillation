from setfit import SetFitModel
import torch
import nltk
import pandas as pd
from transformers import LlamaTokenizer, TFAutoModel
from nltk.translate.bleu_score import sentence_bleu
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy

def calculate_bleu(reference, candidate):
    """
    Calculates the BLEU score for a candidate sentence against a reference sentence.

    Args:
        reference (list): A list of reference sentences.
        candidate (str): The candidate sentence.

    Returns:
        float: The BLEU score.
    """

    return sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))

# Example usage
# reference = [["the", "cat", "is", "on", "the", "mat"]]
# candidate = "the cat sat on the mat"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_path = './teacher-model'
student_path = './distilled-model'
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")

trainset_filename = 'train.csv'
train_df = pd.read_csv(trainset_filename)
train_size = len(train_df) // 50
train_df = train_df.iloc[0:train_size]
print(f"Train Size: {train_size}")
# train_df.rename(columns={'code': 'text', 'docstring': 'label'}, inplace=True)
# train_df['code_tokens'] = train_df['code_tokens'].apply(ast.literal_eval)
# train_df['docstring_tokens'] = train_df['docstring_tokens'].apply(ast.literal_eval)
# train_dataset = Dataset.from_pandas(train_df)

evalset_filename = 'eval.csv'
eval_df = pd.read_csv(evalset_filename)
eval_size = len(eval_df) // 10
eval_df = eval_df.iloc[0:test_size]

testset_filename = 'test.csv'
test_df = pd.read_csv(testset_filename)
test_size = len(test_df) // 100
test_df = test_df.iloc[0:test_size]

teacher_model = SetFitModel.from_pretrained(teacher_path).to(device)

train_embeddings = teacher_model.model_body.encode([row['code'] for _, row in train_df.iterrows()])
teacher_model.model_head = LogisticRegression(multi_class='auto', max_iter=1000)
teacher_model.model_head.fit(train_embeddings, [row['docstring'] for _, row in train_df.iterrows()])

teacher_results = []
total_BLEU_score = 0
num_tests = 0

#Teacher testing
for _, row in test_df.iterrows():

    # model_input = tokenizer(row['code'], return_tensors="pt", truncation=True, max_length=2048).input_ids
    model_input = row['code']
        
    expected_output = row['docstring']
    
    print('Expected: ',expected_output)

    # predicted_output = teacher_model.predict(model_input.cuda())
    predicted_output = teacher_model.predict(model_input)

    # predicted_output = tokenizer.decode(predicted_output[0], skip_special_tokens=True)
    print('Predicted: ',predicted_output)
    
    score = calculate_bleu(expected_output.split(), predicted_output.split())
    print("BLEU Score:", score)

    total_BLEU_score += score
    num_tests += 1

    teacher_results.append({
        'model_input': row['code'],
        'expected_ouput': expected_output,
        'predicted_output': predicted_output,
        'BLEU_score': score,
    })

teacher_results.append({
        'model_input': '',
        'expected_ouput': '',
        'predicted_output': 'Average BLEU score: ',
        'BLEU_score': total_BLEU_score/num_tests,
    })
pd.DataFrame(teacher_results).to_csv('teacher_results.csv', index=False)

student_results = []
total_BLEU_score = 0
num_tests = 0

student_model = SetFitModel.from_pretrained(student_path).to(device)
# train_embeddings = student_model.model_body.encode([row['code'] for _, row in eval_df.iterrows()])
# student_model.model_head = LogisticRegression(multi_class='auto', max_iter=1000)
# student_model.model_head.fit(train_embeddings, [row['docstring'] for _, row in eval_df.iterrows()])

train_embeddings = student_model.model_body.encode([row['code'] for _, row in train_df.iterrows()])
student_model.model_head = LogisticRegression(multi_class='auto', max_iter=1000)
student_model.model_head.fit(train_embeddings, [row['docstring'] for _, row in train_df.iterrows()])

# Student testing
for _, row in test_df.iterrows():
    model_input = row['code']
        
    expected_output = row['docstring']
    
    print('Expected: ',expected_output)

    # predicted_output = teacher_model.predict(model_input.cuda())
    predicted_output = student_model.predict(model_input)

    # predicted_output = tokenizer.decode(predicted_output[0], skip_special_tokens=True)
    print('Predicted: ',predicted_output)
    
    score = calculate_bleu(expected_output.split(), predicted_output.split())
    print("BLEU Score:", score)

    total_BLEU_score += score
    num_tests += 1

    student_results.append({
        'model_input': row['code'],
        'expected_ouput': expected_output,
        'predicted_output': predicted_output,
        'BLEU_score': score,
    })

student_results.append({
        'model_input': '',
        'expected_ouput': '',
        'predicted_output': 'Average BLEU score: ',
        'BLEU_score': total_BLEU_score/num_tests,
    })
pd.DataFrame(student_results).to_csv('student_results.csv', index=False)
