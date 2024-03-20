import pandas as pd
import numpy as np
import pickle
import string
import argparse
import os
import json
import random
import torch
import urllib.request
import pytreebank
import nltk
from nltk.tokenize import sent_tokenize
from glob import glob
from shutil import rmtree
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, logging
from datasets import Dataset, load_metric, concatenate_datasets, load_dataset
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support



def compute_metrics(eval_preds):
    metric = load_metric('accuracy')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(args, train_dataset, val_dataset):
    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # load tokenizer and model
    model_name = args.model_name
    dropout = args.dropout
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, classifier_dropout=dropout)
        
    train_dataset_tokenized = train_dataset.map(preprocess_function, batched=True)
    val_dataset_tokenized = val_dataset.map(preprocess_function, batched=True)
    
    
    # set hyperparameters
    training_args = TrainingArguments(
        output_dir=args.chckpt_dir,
        learning_rate=1e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        warmup_ratio=0,
        evaluation_strategy='epoch',  # Evaluate at the end of each epoch
        save_strategy='epoch',  # Save a checkpoint at the end of each epoch
        logging_strategy='epoch',  # Log at the end of each epoch
        num_train_epochs=5,  # Train for exactly five epochs
        seed=seed,
        optim='adamw_hf',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        fp16=True,
        save_total_limit=5  # Keep all five checkpoints; adjust as needed
        # Removed max_steps
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Add the callback here
    )

    logging.set_verbosity_error()
    train_output = trainer.train()
    print(train_output)
    
    model.save_pretrained("best_roberta")
    tokenizer.save_pretrained("best_roberta")

def eval(args, test_dataset):
    model_name = args.model_name
    output_dir = args.chckpt_dir + 'checkpoint-4455'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,  # For loading the model
        per_device_eval_batch_size=64,
        seed=args.seed,  # Ensure reproducibility
        fp16=True,  # If using mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )
    
    # Convert the pandas DataFrame to Hugging Face Dataset
    # test_dataset = Dataset.from_pandas(test_dataset)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    # Make predictions
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids

    # Compute metrics
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    # Print metrics
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Append metrics to a result.txt file
    with open("result.txt", "a") as file:
        file.write(f"Roberta\n")
        file.write(f"Accuracy: {acc}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1: {f1}\n")




def get_data():
    train_dataset = Dataset.from_dict(pd.read_csv('data/train.csv'))
    dev_dataset = Dataset.from_dict(pd.read_csv('data/dev.csv'))
    test_dataset = Dataset.from_dict(pd.read_csv('data/test.csv'))
    
    
    return train_dataset, dev_dataset, test_dataset


    

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the model")
    
    # Model configuration arguments
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Name of the model to use')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for the model')
    parser.add_argument('--chckpt_dir', type=str, default='checkpoints/', help='route to save/load model')

    # Action arguments
    parser.add_argument('--do_train', action='store_true', help='Train the model')
    parser.add_argument('--do_eval', action='store_true', help='Evaluate the model')

    args = parser.parse_args()
    
    train_dataset, val_dataset, test_dataset = get_data()
    
    if args.do_train:
        # train_dataset, val_dataset = get_train_data(args)
        train(args, train_dataset, val_dataset)

    if args.do_eval:
        eval(args, test_dataset)


if __name__ == "__main__":
    main()












