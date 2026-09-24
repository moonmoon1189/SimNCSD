import torch
import torch.nn as nn
from transformers import AutoModel


class SimNCSDEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Shared Dual-Tower RoBERTa
        self.roberta = AutoModel.from_pretrained(config['pretrained_model_name_or_path'])
        self.dropout = nn.Dropout(config['dropout'])
        self.projection = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['projection_dim'])
        )

    def forward(self, input_ids, attention_mask, return_projection=True):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Take [CLS] token representation
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)

        if return_projection:
            proj_rep = self.projection(cls_rep)
            return cls_rep, proj_rep
        return cls_rep


class MultiChannelFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source_rep, comment_rep):
        # f = [v_source, v_comment, |v_source - v_comment|, v_source * v_comment]
        abs_diff = torch.abs(source_rep - comment_rep)
        hadamard = source_rep * comment_rep
        fusion = torch.cat([source_rep, comment_rep, abs_diff, hadamard], dim=-1)
        return fusion


class SimNCSDClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['hidden_size'] * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config['mlp_hidden_1']),
            nn.ReLU(),
            nn.Linear(config['mlp_hidden_1'], config['mlp_hidden_2']),
            nn.ReLU(),
            nn.Linear(config['mlp_hidden_2'], 1),
            nn.Sigmoid()
        )

    def forward(self, fused_features):
        return self.mlp(fused_features).squeeze(-1)