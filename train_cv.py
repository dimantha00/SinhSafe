import pandas as pd
import numpy as np
import torch
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

# --- Configuration ---
MODEL_NAME = "xlm-roberta-large"
DATA_DIR = r"data/processed_ground_truth"
N_FOLDS = 5  # Standard for research

id2label = {0: "Normal", 1: "Offensive", 2: "Cyberbullying"}
label2id = {"Normal": 0, "Offensive": 1, "Cyberbullying": 2}

# --- Load Data ---
def load_data():
    df_bully = pd.read_excel(os.path.join(DATA_DIR, "processed_cyberbullying.xlsx"))
    df_offen = pd.read_excel(os.path.join(DATA_DIR, "processed_offensive.xlsx"))
    df_norm = pd.read_excel(os.path.join(DATA_DIR, "processed_normal.xlsx"))

    df_norm['label'] = 0
    df_offen['label'] = 1
    df_bully['label'] = 2

    df = pd.concat([df_bully, df_offen, df_norm], ignore_index=True)
    df = df[['cleaned_text', 'label']].dropna()
    df.rename(columns={'cleaned_text': 'text'}, inplace=True)
    df['text'] = df['text'].astype(str)
    return df

df = load_data()
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Cross Validation Loop ---
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = []

print(f"Starting {N_FOLDS}-Fold Cross-Validation on RTX 3090...")

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    # 1. Split Data for this Fold
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # 2. Re-Initialize Model (Critical: Must start fresh each fold)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id
    ).to("cuda")
    
    # 3. Training Args
    training_args = TrainingArguments(
        output_dir=f'./results/fold_{fold}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True, # RTX 3090 Speed boost
        dataloader_num_workers=4,
        report_to="none" # Keep console clean
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # 4. Train
    trainer.train()
    
    # 5. Evaluate
    eval_result = trainer.evaluate()
    print(f"Fold {fold+1} Result: Accuracy={eval_result['eval_accuracy']:.4f}")
    fold_results.append(eval_result['eval_accuracy'])
    
    # 6. Save the Best Model from this fold (Optional, but good to keep one)
    # If this is the best fold so far, save it as the 'master' model
    if eval_result['eval_accuracy'] == max(fold_results):
        print(">> Current Best Model! Saving to 'models/best_cv_model'...")
        model.save_pretrained("models/best_cv_model")
        tokenizer.save_pretrained("models/best_cv_model")

    # 7. Cleanup VRAM
    del model, trainer, training_args
    gc.collect()
    torch.cuda.empty_cache()

# --- Final Report ---
print("\n" + "="*30)
print(f"Final Results ({N_FOLDS}-Fold CV)")
print("="*30)
for i, acc in enumerate(fold_results):
    print(f"Fold {i+1}: {acc:.4f}")
print(f"\nAverage Accuracy: {np.mean(fold_results):.4f}")
print("="*30)