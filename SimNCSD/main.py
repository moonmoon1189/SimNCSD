import sys
import os
# 将当前 main.py 所在的根目录强制加入 Python 搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import yaml
import torch
import warnings
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.data.dataset import SimNCSDDataset
from src.models.simncsd import SimNCSDEncoder, MultiChannelFusion, SimNCSDClassifier
from src.training.train_contrastive import train_stage_1
from src.training.train_classifier import train_stage_2
from src.evaluation.evaluate import evaluate_metrics
from src.visualization.plot_results import plot_and_save_results

warnings.filterwarnings("ignore")


def main():
    # 1. Load config and set seed
    with open('configs/base.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    set_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer_name = config['model']['pretrained_model_name_or_path']
    max_length = config['model']['max_length']

    # 2. Load Datasets
    print("Loading datasets...")
    train_dataset_stage1 = SimNCSDDataset('data/processed/train.jsonl', tokenizer_name, max_length, is_contrastive=True)
    train_dataset_stage2 = SimNCSDDataset('data/processed/train.jsonl', tokenizer_name, max_length,
                                          is_contrastive=False)
    val_dataset = SimNCSDDataset('data/processed/val.jsonl', tokenizer_name, max_length, is_contrastive=False)
    test_dataset = SimNCSDDataset('data/processed/test.jsonl', tokenizer_name, max_length, is_contrastive=False)

    # 3. Initialize Models
    print("Initializing SimNCSD network components...")
    encoder = SimNCSDEncoder(config['model']).to(device)
    fusion_layer = MultiChannelFusion().to(device)
    classifier = SimNCSDClassifier(config['model']).to(device)

    # 4. Stage 1: Contrastive Fine-Tuning
    print("\n" + "=" * 50)
    print("STAGE 1: Supervised Cross-Text Contrastive Fine-Tuning")
    print("=" * 50)
    encoder = train_stage_1(encoder, train_dataset_stage1, config['training'], device)

    # 5. Stage 2: Classifier Training
    print("\n" + "=" * 50)
    print("STAGE 2: Multi-Channel Feature Fusion & Classifier Training")
    print("=" * 50)
    classifier, optimal_threshold = train_stage_2(
        encoder, classifier, fusion_layer, train_dataset_stage2, val_dataset, config['training'], device
    )

    # 6. Evaluation on Test Set
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 50)
    encoder.eval()
    classifier.eval()
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    test_probs = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            source_input = batch['source_input_ids'].to(device)
            source_mask = batch['source_attention_mask'].to(device)
            comment_input = batch['comment_input_ids'].to(device)
            comment_mask = batch['comment_attention_mask'].to(device)
            labels = batch['label'].to(device)

            source_rep, _ = encoder(source_input, source_mask, return_projection=False)
            comment_rep, _ = encoder(comment_input, comment_mask, return_projection=False)
            fused_feats = fusion_layer(source_rep, comment_rep)
            preds = classifier(fused_feats)

            test_probs.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    final_metrics = evaluate_metrics(test_labels, test_probs, threshold=optimal_threshold)

    print("\nSimNCSD Final Test Metrics:")
    print(f"Validation Calibrated Threshold (gamma): {optimal_threshold:.4f}")
    print(f"Accuracy:  {final_metrics['Accuracy'] * 100:.2f}%")
    print(f"Precision: {final_metrics['Precision'] * 100:.2f}%")
    print(f"Recall:    {final_metrics['Recall'] * 100:.2f}%")
    print(f"F1-Score:  {final_metrics['F1-Score'] * 100:.2f}%")
    print(f"AUC:       {final_metrics['AUC']:.4f}")

    # 7. Generate and save plots
    print("\nGenerating visualization plots...")
    plot_and_save_results(test_labels, test_probs, optimal_threshold, output_dir="outputs/figures")


if __name__ == "__main__":
    main()