import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from src.evaluation.threshold import calibrate_threshold_youden
from src.evaluation.evaluate import evaluate_metrics


def train_stage_2(encoder_model, classifier_model, fusion_layer, train_dataset, val_dataset, config, device):
    """Stage 2: Classifier Training with Frozen Encoder"""
    encoder_model.eval()  # Freeze encoder
    classifier_model.to(device)
    classifier_model.train()

    for param in encoder_model.parameters():
        param.requires_grad = False

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    optimizer = AdamW(classifier_model.parameters(), lr=config['learning_rate_classifier'],
                      weight_decay=config['weight_decay'])
    criterion = nn.BCELoss()

    best_val_auc = 0
    patience_counter = 0
    optimal_threshold = 0.5

    for epoch in range(config['epochs_classifier']):
        classifier_model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch + 1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            source_input = batch['source_input_ids'].to(device)
            source_mask = batch['source_attention_mask'].to(device)
            comment_input = batch['comment_input_ids'].to(device)
            comment_mask = batch['comment_attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                # get original 768-dim pooled representations (drop projection)
                source_rep, _ = encoder_model(source_input, source_mask, return_projection=False)
                comment_rep, _ = encoder_model(comment_input, comment_mask, return_projection=False)

            fused_feats = fusion_layer(source_rep, comment_rep)
            preds = classifier_model(fused_feats)

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        # Validation Phase
        classifier_model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                source_input = batch['source_input_ids'].to(device)
                source_mask = batch['source_attention_mask'].to(device)
                comment_input = batch['comment_input_ids'].to(device)
                comment_mask = batch['comment_attention_mask'].to(device)
                labels = batch['label'].to(device)

                source_rep, _ = encoder_model(source_input, source_mask, return_projection=False)
                comment_rep, _ = encoder_model(comment_input, comment_mask, return_projection=False)
                fused_feats = fusion_layer(source_rep, comment_rep)
                preds = classifier_model(fused_feats)

                val_probs.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calibrate threshold and evaluate
        current_threshold = calibrate_threshold_youden(val_labels, val_probs)
        val_metrics = evaluate_metrics(val_labels, val_probs, current_threshold)
        print(
            f"Validation AUC: {val_metrics['AUC']:.4f} | F1: {val_metrics['F1-Score']:.4f} | Threshold: {current_threshold:.4f}")

        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
            optimal_threshold = current_threshold
            patience_counter = 0
            # Save model checkpoit here if needed
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print("Early stopping triggered.")
                break

    return classifier_model, optimal_threshold