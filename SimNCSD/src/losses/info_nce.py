import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedCrossTextInfoNCE(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, source_reps, pos_reps, neg_reps):
        """
        source_reps: (batch_size, dim)
        pos_reps: (batch_size, dim)
        neg_reps: (batch_size, dim) - explicit hard negatives
        """
        # Normalize representations
        source_reps = F.normalize(source_reps, p=2, dim=1)
        pos_reps = F.normalize(pos_reps, p=2, dim=1)
        neg_reps = F.normalize(neg_reps, p=2, dim=1)

        batch_size = source_reps.size(0)

        # Calculate cosine similarities
        # Positive pairs similarity: (batch_size, 1)
        sim_pos = torch.sum(source_reps * pos_reps, dim=-1, keepdim=True) / self.temperature

        # In-batch implicit negatives: (batch_size, batch_size)
        sim_in_batch = torch.matmul(source_reps, pos_reps.T) / self.temperature

        # Explicit hard negatives: (batch_size, batch_size)
        sim_hard_neg = torch.matmul(source_reps, neg_reps.T) / self.temperature

        # Mask out the diagonal for in-batch negatives (since they are the positive pairs)
        mask = torch.eye(batch_size, device=source_reps.device).bool()
        sim_in_batch.masked_fill_(mask, -1e9)

        # Denominator combines pos, in-batch negatives, and explicit hard negatives
        logits = torch.cat([sim_pos, sim_in_batch, sim_hard_neg], dim=1)

        # Labels are 0 because the positive pair is at index 0 for each row
        labels = torch.zeros(batch_size, dtype=torch.long, device=source_reps.device)

        loss = F.cross_entropy(logits, labels)
        return loss