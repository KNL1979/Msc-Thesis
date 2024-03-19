# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:11:24 2024

@author: kimlu
"""

import csv

# Open the TSV file
with open('data/test_en.tsv', newline='', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')  # Specify tab ('\t') as the delimiter
    for row in reader:
        print(row)  # Print each row

#%%
import csv
import re

def read_dataset(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        sentence = []
        label = []
        for row in reader:
            if row and len(row) >= 2:  # Check if the row is not empty and has at least two columns
                sentence.append(row[1])  # Assuming the sentence is in the second column
                label_text = row[-1]
                label.append(extract_label(label_text))  # Extract label
            else:
                if sentence:  # Append the last sentence if available
                    sentences.append(sentence)
                    labels.append(label)
                sentence = []
                label = []
        if sentence:  # Append the last sentence and label (in case the file doesn't end with an empty row)
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

def extract_label(label_text):
    # Use regular expression to extract label
    match = re.search(r'\b[A-Z]+-[A-Z]+\b', label_text)
    if match:
        return match.group()
    else:
        return 'O'  # Return 'O' if no label is found

# File path for the train set
train_file_path = 'data/train_en.tsv'

# Preprocess the train set
train_sentences, train_labels = read_dataset(train_file_path)

# Display the first few sentences and corresponding labels
for i in range(5):  # Displaying the first 5 examples
    print("Labels:", ", ".join(train_labels[i]))  # Join the labels into a comma-separated string
    print("Sentence:", " ".join(train_sentences[i]))  # Join the sentence tokens into a single string
    print()


#%% Train own model on the multinerd dataset

from transformers import TrainingArguments, Trainer
import numpy as np

# Replace this with your own 'my_NER' model initialization
model = my_NER.from_pretrained("path_to_your_model_checkpoint")

# Replace the model_name with your own model name if necessary
model_name = "my_NER"

# Adjust the training arguments as needed
args = TrainingArguments(
    "my_NER-training",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
    eval_steps=10000,
    save_steps=10000,
)

# Load the Seqeval metric
metric = load_metric("seqeval")

# Define a function to compute evaluation metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Initialize the Trainer with your 'my_NER' model, training arguments, data collator, tokenizer, and compute metrics function
trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Evaluate the trained model
trainer.evaluate()

# Make predictions and compute metrics
predictions, labels, _ = trainer.predict(test_tokenized)
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results










