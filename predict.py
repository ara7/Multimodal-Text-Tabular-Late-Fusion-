# Author: Ara, Lena
# Description: Script for evaluating the trained model on the test set.
# Loads a model checkpoint and test data, then computes and prints
# key performance metrics like AUC, F1-score, and confusion matrix.

import pandas as pd
import numpy as np
import random
import torch
import os
import argparse  # Used to specify which model checkpoint to load
import plotly.express as px

import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer

import config
from dataset import MultimodalDataset
from model import BertClassifier

from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_fscore_support
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def per_class_accuracy(y_true, y_pred, class_of_interest):
    """Calculates accuracy for a specific class."""
    true_class_indices = (y_true == class_of_interest)
    true_positives = np.sum((y_true[true_class_indices] == y_pred[true_class_indices]))
    total_instances = np.sum(true_class_indices)
    per_class_acc = true_positives / total_instances if total_instances > 0 else 0.0
    return per_class_acc, true_positives

def evaluate_roc(probs, y_true):
    """
    - Prints AUC, classification report, and other metrics.
    - Generates and saves a Plotly ROC curve.
    """
    preds_proba = probs[:, 1]  # Probability of the positive class
    preds_binary = np.where(preds_proba >= 0.5, 1, 0)

    # --- Calculate Metrics ---
    roc_auc = roc_auc_score(y_true, preds_proba)
    accuracy = accuracy_score(y_true, preds_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds_binary, average='binary')

    print("\n======== Evaluation Report ========")
    print(f"  AUC: {roc_auc:.4f}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (Sensitivity): {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, preds_binary))

    # --- Per-Class Accuracy ---
    acc_class_1, tp_1 = per_class_accuracy(y_true, preds_binary, class_of_interest=1)
    acc_class_0, tp_0 = per_class_accuracy(y_true, preds_binary, class_of_interest=0)

    print("\n--- Per-Class Details ---")
    print(f"  Class 1 (Positive) Accuracy: {acc_class_1:.4f} (TP: {tp_1})")
    print(f"  Class 0 (Negative) Accuracy: {acc_class_0:.4f} (TN: {tp_0})")

    # --- Confusion Matrix ---
    conf_matrix = confusion_matrix(y_true, preds_binary)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n--- Confusion Matrix ---")
    print(conf_matrix)
    print(f"  True Positives (TP): {tp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Specificity: {specificity:.4f}")
    print("========================================")

    # --- Plot ROC AUC ---
    fpr, tpr, thresholds = roc_curve(y_true, preds_proba)

    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})
    fig = px.area(
        roc_df,
        x="FPR",
        y="TPR",
        hover_data=["Threshold"],
        title=f"ROC Curve (AUC = {roc_auc:.4f})"
    )
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(width=700, height=500)

    # Save plot as an interactive HTML file
    plot_filename = "roc_curve_test.html"
    fig.write_html(plot_filename)
    print(f"\nSaved interactive ROC plot to {plot_filename}")


def bert_predict(model, test_dataloader):
    """Performs inference on the test set."""

    model.eval()
    all_logits = []
    all_labels = []
    all_conf_flags = []
    all_sids = []

    print("\nRunning predictions on test set...")
    for batch in test_dataloader:
        # Move batch to device
        b_input_ids = batch['input_ids'].to(device)
        b_attn_mask = batch['attention_mask'].to(device)
        b_tabular = batch['tabular_data'].to(device)
        b_labels = batch['label'].to(device)

        with torch.no_grad():
            logits, conf_flag, sid = model(b_input_ids, b_attn_mask, b_tabular)

        all_logits.append(logits)
        all_labels.append(b_labels)
        all_conf_flags.append(conf_flag)
        all_sids.append(sid)

    # Concatenate all batch results
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    all_conf_flags = torch.cat(all_conf_flags, dim=0).cpu().numpy()
    all_sids = torch.cat(all_sids, dim=0).cpu().numpy()

    # Calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    # Create predictions dataframe
    preds_binary = np.argmax(probs, axis=1)
    df_preds = pd.DataFrame({
        'SID': all_sids,
        'confusion_flag': all_conf_flags,
        'true_label': all_labels,
        'predicted_label': preds_binary,
        'probability_class_0': probs[:, 0],
        'probability_class_1': probs[:, 1]
    })

    pred_filename = "test_predictions.csv"
    df_preds.to_csv(pred_filename, index=False)
    print(f"Saved predictions to {pred_filename}")

    return probs, all_labels

def main(args):
    """Main execution function for prediction."""

    set_seed(config.SEED)

    # --- 1. Load Test Data ---
    print(f"Loading test data from {config.TEST_DATA}...")
    try:
        df_test = pd.read_csv(config.TEST_DATA).dropna(subset=[config.TEXT_FEATURE])
        y_test_true = df_test[config.LABEL_COLUMN].to_numpy()
        print(f"Loaded {len(df_test)} test samples.")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {config.TEST_DATA}")
        return

    # --- 2. Create Dataset and DataLoader ---
    print("Initializing tokenizer and test dataset...")
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME, do_lower_case=True)

    test_dataset = MultimodalDataset(
        df=df_test,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset), # No shuffling for test set
        batch_size=config.BATCH_SIZE
    )

    # --- 3. Load Trained Model ---
    print(f"Loading model from checkpoint: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        print("Please train a model first using train.py")
        return

    # Initialize the model architecture
    model = BertClassifier()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    test_probs, y_true = bert_predict(model, test_dataloader)

    assert np.array_equal(y_true, y_test_true), "Label mismatch between CSV and DataLoader!"

    evaluate_roc(test_probs, y_true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the multimodal classifier on the test set.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pth file) to use for prediction."
    )
    args = parser.parse_args()

    main(args)
