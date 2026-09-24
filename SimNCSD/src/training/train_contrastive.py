import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm


def train_stage_1(model, dataset, config, device):
    """Stage 1: Supervised Cross-Text Contrastive Fine-Tuning"""
    model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate_contrastive'], weight_decay=config['weight_decay'])

    total_steps = len(dataloader) * config['epochs_contrastive']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config['warmup_ratio']),
        num_training_steps=total_steps
    )

    from src.losses.info_nce import SupervisedCrossTextInfoNCE
    criterion = SupervisedCrossTextInfoNCE(temperature=config['temperature'])

    for epoch in range(config['epochs_contrastive']):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Stage 1 - Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            source_input = batch['source_input_ids'].to(device)
            source_mask = batch['source_attention_mask'].to(device)
            pos_input = batch['pos_input_ids'].to(device)
            pos_mask = batch['pos_attention_mask'].to(device)
            neg_input = batch['neg_input_ids'].to(device)
            neg_mask = batch['neg_attention_mask'].to(device)

            _, source_proj = model(source_input, source_mask)
            _, pos_proj = model(pos_input, pos_mask)
            _, neg_proj = model(neg_input, neg_mask)

            loss = criterion(source_proj, pos_proj, neg_proj)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch + 1} Average Loss: {total_loss / len(dataloader):.4f}")

    return model