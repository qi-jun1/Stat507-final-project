# Stat507-final-project
from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb")
print(ds)  

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_ds = ds.map(tokenize_function, batched=True)
print(tokenized_ds["train"][0])

split_ds = tokenized_ds["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split_ds["train"]
val_ds = split_ds["test"]

test_ds = tokenized_ds["test"]

from transformers import TrainingArguments, Trainer
import numpy as np

import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",              
    evaluation_strategy="epoch",         
    learning_rate=2e-5,                  
    per_device_train_batch_size=16,      
    per_device_eval_batch_size=16,       
    num_train_epochs=3,                  
    weight_decay=0.01,                   
    logging_dir='./logs',                
    logging_steps=100,                  
)

trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=train_ds,              
    eval_dataset=val_ds,                 
    compute_metrics=compute_metrics,     
)

trainer.train()

results = trainer.evaluate(test_ds)
print("Test Evaluation Results:", results)
