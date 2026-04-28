import os 
import copy
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import lightning.pytorch as L
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections.abc import Iterable
from torch_scatter import scatter_softmax
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, 
    precision_recall_curve, roc_curve, accuracy_score
)
from sklearn.isotonic import IsotonicRegression

from interpro import InterProData

os.chdir(os.path.expanduser("~"))
selected_indices = np.load("code/selected_feature_indices.npy")
with open("data/uniprotkb_index_mapping.json", 'r') as f:
    index_mapping = json.load(f)

def fmax(targets, logits):
    """
    Calculate the maximum F1-score (Fmax) for multi-class classification.
    :param targets: Ground truth labels (batch_size, num_classes)
    :param logits: Predicted scores (batch_size, num_classes)
    :return: Maximum F1-score (Fmax) and the corresponding optimal threshold.
    """
    # For multi-class, consider one-vs-rest (one class at a time)
    fmax_scores = []  # List to store Fmax for each class
    optimal_thresholds = []  # List to store the optimal threshold for each class

    if logits.ndim == 1:
        logits = logits[:, np.newaxis]  # (batch_size,) -> (batch_size, 1)
        targets = targets[:, np.newaxis]

    num_classes = logits.shape[1]

    for i in range(num_classes):  # Iterate over each class
        y_true = targets[:, i]    # One-vs-rest for each class
        y_score = logits[:, i]  # The predicted score for the i-th class

        # Compute Precision-Recall curve for each class
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores = np.nan_to_num(f1_scores)  # Remove NaN values (if any)
        
        # Find the maximum F1 score (Fmax) for this class
        fmax_scores.append(np.max(f1_scores))  # Store the maximum F1 score
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        optimal_thresholds.append(optimal_threshold)

    # Calculate the average Fmax across all classes
    return np.mean(fmax_scores), optimal_thresholds

def aupr(targets, logits):
    """
    Calculate the AUPR (Area Under Precision-Recall curve) for multi-class classification.
    :param targets: Ground truth labels (batch_size, num_classes)
    :param logits: Predicted scores (batch_size, num_classes)
    :return: AUPR value
    """
    # Calculate AUPR using average precision score
    # For multi-class, compute AUPR for each class and return the weighted average
    aupr_scores = []
    if logits.ndim == 1:
        logits = logits[:, np.newaxis]  # (batch_size,) -> (batch_size, 1)
        targets = targets[:, np.newaxis]
    num_classes = logits.shape[1]

    for i in range(num_classes):  # Iterate over each class
        y_true = targets[:, i] 
        y_score = logits[:, i]  # The predicted score for the i-th class

        # Calculate average precision (AP) for each class (AUPR)
        ap = average_precision_score(y_true, y_score)
        aupr_scores.append(ap)
    
    # Return the average AUPR across all classes
    return np.mean(aupr_scores)


class IdentityWithArgs(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TopkRouter(nn.Module):
    def __init__(self, in_channels, num_experts, top_k, noise=False):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.noise = noise
        self.topkroute_linear = nn.Linear(in_channels, num_experts)
        self.noise_linear =nn.Linear(in_channels, num_experts)
    
    def forward(self, multi_output, return_moe=False):
        logits = self.topkroute_linear(multi_output)

        if self.noise:
            noise_logits = self.noise_linear(multi_output)   
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise
        
        # Retrieve the top K values and their indices, column by column.
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        # Create a zero matrix with the same shape as logits, filled with -inf
        zeros = torch.full_like(logits, float('-inf'))
        # Scatter the top K logits into the zero matrix at the corresponding indices, resulting in a sparse matrix where only the top K positions have their original logits and the rest are -inf
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # Apply softmax to the sparse logits to get the router output, which will have non-zero probabilities only at the top K positions and zero elsewhere
        router_output = F.softmax(sparse_logits, dim=-1)

        router_stats = {
            "router_logits": None,
            "router_output": None,
            "indices": None,
        }
        if return_moe:
            router_stats = {
                "router_logits": logits.detach().cpu(),
                "router_output": router_output.detach().cpu(),
                "indices": indices.detach().cpu(),
            }
        return router_output, indices, router_stats


class SparseMoE(nn.Module):
    def __init__(self, in_channels, num_experts, top_k, Experts,
                 noise=False, return_moe=False):
        super(SparseMoE, self).__init__()
        self.top_k = top_k
        self.router = TopkRouter(in_channels, num_experts, top_k, noise)
        self.experts = nn.ModuleList([Experts for _ in range(num_experts)])
        self.return_moe = return_moe
        self.latest_moe_stats = None

    def forward(self, x):
        moe_stats = {
            "router": {
                "router_logits": None,
                "gating_output": None,
                "indices": None,
            },
            "experts": {}
        }
        original_shape = x.shape  # Record the original shape
        if x.dim() == 2:  # Case without batch dimension, shape [L, d]
            x = x.unsqueeze(0)  # Add batch dimension, become [1, L, d]

        # 1. Input goes through the router to get two outputs
        gating_output, indices, router_stats = self.router(x, return_moe=self.return_moe)
        moe_stats["router"] = router_stats
        # 2. Initialize a zero matrix, which will be accumulated to the final result
        final_output = torch.zeros_like(x)

        # 3. Flatten, concatenate each batch together, flatten both input x and router output
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Operate on each expert, i.e., weight all tokens processed by the current expert
        for i, expert in enumerate(self.experts):
            # 4. For the current expert (e.g., expert 0), see which tokens are in the top k
            expert_mask = (indices == i).any(dim=-1)
            # 5. Flatten operation
            flat_mask = expert_mask.view(-1)
            # If the current expert is in the top k for any token
            if flat_mask.any():
                # 6. Get the tokens that the expert acts on, select the token dimension
                expert_input = flat_x[flat_mask]
                # 7. Input the tokens into the expert to get the output
                expert_output = expert(expert_input)

                # 8. Calculate the gating scores for the tokens that the expert acts on
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # 9. Multiply the expert output by the gating scores
                weighted_output = expert_output * gating_scores
                # 10. Accumulate the results in a loop
                final_output[expert_mask] += weighted_output

                if self.return_moe:
                    moe_stats["experts"][i] = {
                        "num_tokens": int(flat_mask.sum().item()),
                        "mean_gate": float(gating_scores.mean().item()),
                        "max_gate": float(gating_scores.max().item()),
                    }

        self.latest_moe_stats = moe_stats
                
        if len(original_shape) == 2:
            final_output = final_output.squeeze(0)  # Restore to [L, d]

        return final_output


class FFN(nn.Module):
    def __init__(self, in_channels, activation=True):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU() if activation else nn.Identity(),
            nn.Linear(in_channels * 2, in_channels)
        )

    def forward(self, x):
        return self.ffn(x)
    

class Attention(nn.Module):
    def __init__(self, in_channels, head=1, num_experts=4, top_k=2, return_analysis=False):
        super().__init__()
        assert in_channels % head == 0, "in_channels must be divisible by head"
        self.head = head
        hidden_channels = in_channels // head
        self.d = torch.tensor(hidden_channels, dtype=torch.float32)
        self.linear_Q = nn.ModuleList([nn.Linear(in_channels, hidden_channels, bias=False) for _ in range(head)])
        self.linear_K = nn.ModuleList([nn.Linear(in_channels, hidden_channels, bias=False) for _ in range(head)])
        self.linear_V = nn.ModuleList([nn.Linear(in_channels, hidden_channels, bias=False) for _ in range(head)])

        self.linear_cat = nn.Linear(hidden_channels * head, in_channels)
        self.layernorm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.3) 
        ffn = FFN(in_channels)# , activation=False
        self.ffn = SparseMoE(in_channels, num_experts, top_k, Experts=ffn)

        self.return_analysis = return_analysis

    def forward(self, x, g, batch, bias=None, lambda_=0.5):
        g_expanded = g[batch]
        multi_output = []
        analysis = {
            "heads": {},
            "bias_raw": bias
        }
        # analysis = {"raw_scores": [], "bias_applied_scores": [], "alphas": [], "bias_raw": bias}
        for i in range(self.head):
            analysis["heads"][i] = {}
            Q = self.linear_Q[i](g_expanded) 
            K = self.linear_K[i](x)
            V = self.linear_V[i](x)
            scores = torch.sum(torch.mul(Q, K)/torch.sqrt(self.d), dim=1, keepdim=True)
            if self.return_analysis:
                # analysis["raw_scores"].append(scores.detach().cpu())
                analysis["heads"][i]["raw_scores"] = scores.detach().cpu()
            if bias is not None:
                bias_clipped = bias.clamp(max=5.0)
                scores = scores * (1 + lambda_ * bias_clipped)
                if self.return_analysis:
                    # analysis["bias_applied_scores"].append(scores.detach().cpu())
                    analysis["heads"][i]["bias_applied_scores"] = scores.detach().cpu()
            alpha = scatter_softmax(scores.squeeze(-1), batch).unsqueeze(-1)  # shape: (L, 1)
            if self.return_analysis:
                # analysis["alphas"].append(alpha.detach().cpu())
                analysis["heads"][i]["alphas"] = alpha.detach().cpu()
            updated_features = V * alpha  # shape: (L, d)
            multi_output.append(updated_features)
            
        multi_output = torch.cat(multi_output, dim=-1)
        multi_output = self.linear_cat(multi_output)
        multi_output = self.dropout(multi_output)
        multi_output = self.layernorm(multi_output + x)
        multi_output = self.dropout(multi_output)
        multi_output = self.layernorm(self.ffn(multi_output)+multi_output)

        return multi_output, analysis
    

class AttentionWithDomainGuide(nn.Module):
    def __init__(self, in_channels, domain_channels=5, num_experts=4, top_k=2):
        super().__init__()
        self.num_heads = domain_channels  # Activation of each domain serves as a trigger
        self.hidden_dim = in_channels // self.num_heads
        self.d = torch.tensor(self.hidden_dim, dtype=torch.float32)

        self.q_proj = nn.Linear(in_channels, self.hidden_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(1, self.hidden_dim)  # Each domain is a scalar, so input dim is 1
        self.v_proj = nn.Linear(1, self.hidden_dim)

        self.out_proj = nn.Linear(self.hidden_dim * self.num_heads, in_channels)
        self.layernorm1 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.3)
        
        ffn = FFN(in_channels)
        self.ffn = SparseMoE(in_channels, num_experts, top_k, Experts=ffn)
        self.layernorm2 = nn.LayerNorm(in_channels)

    def forward(self, x, domain_feats, batch):
        # x: [L, in_channels]
        # domain_feats: [L, 5]
        L = x.size(0)

        Q = self.q_proj(x)  # [L, H*d]
        Q = Q.view(L, self.num_heads, self.hidden_dim)  # [L, 5, d]

        domain_split = domain_feats.split(1, dim=-1)  # 5 x [L, 1]

        outputs = []
        for i, d_i in enumerate(domain_split):  # Iterate over each domain feature
            K = self.k_proj(d_i)  # [L, d]
            V = self.v_proj(d_i)  # [L, d]
            q_i = Q[:, i, :]      # [L, d]

            score = torch.sum(q_i * K, dim=-1, keepdim=True) / torch.sqrt(self.d)  # [L, 1]
            alpha = scatter_softmax(score.squeeze(-1), batch).unsqueeze(-1)  # [L, 1]
            updated = V * alpha  # [L, d]
            outputs.append(updated)

        concat_output = torch.cat(outputs, dim=-1)  # [L, H*d]
        fused = self.out_proj(concat_output)  # [L, in_channels]

        fused = self.dropout(fused)
        fused = self.layernorm1(fused + x)

        fused2 = self.ffn(fused)
        fused2 = self.dropout(fused2)
        out = self.layernorm2(fused2 + fused)
        return out


class ResidualGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.residual_projection = nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))

    def forward(self, x, edge_index):
        residual = self.residual_projection(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = x + residual
            residual = x
        return x


class MLP(nn.Module):
    def __init__(self, n_features, in_channels):
        super().__init__()
        mlp = FFN(n_features, activation=False)
        self.moe = SparseMoE(n_features, num_experts=4, top_k=2, Experts=mlp)
        self.linear = nn.Linear(n_features, in_channels)
    
    def forward(self, x):
        x, _ = self.moe(x)
        return self.linear(x)
    

class SharedEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class TaskClassifier(nn.Module):
    def __init__(self, return_moe=False):
        super().__init__()
        ffn = FFN(128)
        self.net = nn.Sequential(
            SparseMoE(in_channels=128, num_experts=16, top_k=4,
                      Experts=ffn, return_moe=return_moe),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.BatchNorm1d(32),
        )

    def forward(self, x):
        return self.net(x)
    


class Classify(nn.Module):
    def __init__(self, in_channels, hidden_channels, task_names, return_moe=False):
        super().__init__()
        self.return_moe = return_moe
        self.shared_layers = SharedEncoder(in_channels + hidden_channels)
        self.classifiers = nn.ModuleDict({
            task: nn.Sequential(
                TaskClassifier(return_moe=return_moe),  # feature extractor
                nn.Linear(32, 1)                        # final task-specific head
            )
              for task in task_names
        }) 

    def forward(self, x):
        outputs = {}
        shared_out = self.shared_layers(x)
        if self.return_moe:
            self.latest_moe_stats = {}
        else:
            self.latest_moe_stats = None
        for task, classifier in self.classifiers.items():
            out = classifier(shared_out)
            outputs[task] = out

            if self.return_moe:
                moe = get_moe_stats_from_sequential(classifier)
                if moe is not None:
                    self.latest_moe_stats[task] = moe
        # outputs = {task: classifier(shared_out) for task, classifier in self.classifiers.items()}
        return outputs
    
def get_moe_stats_from_sequential(seq):
    for m in seq.modules():
        if isinstance(m, SparseMoE):
            return m.latest_moe_stats
    return None

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, 
                 head=4, attn_layers=2, use_domain=False, use_sae=False,
                 attn_order="sae_first", return_analysis=False):
        super().__init__()
        self.use_domain = use_domain
        self.use_sae = use_sae
        self.attn_order = attn_order
        self.gcn = ResidualGCN(in_channels, hidden_channels, num_layers)
        # SAE attention
        self.sae = (
            AttentionWithDomainGuide(hidden_channels, domain_channels=5)
            if use_sae else IdentityWithArgs()
        )

        # domain attention
        self.mlp = (
            nn.Linear(hidden_channels, hidden_channels)
            if use_domain else IdentityWithArgs()
        )
        self.domain = (
            nn.ModuleList([
                Attention(hidden_channels, head, return_analysis=return_analysis) for _ in range(attn_layers)
            ]) if use_domain else nn.ModuleList([IdentityWithArgs()])
        )
    
    def forward(self, x, edge_index, inter_features, batch, acts_topk=None):
        ana = {}
        x = self.gcn(x, edge_index)
        x_i = x.clone()
        inter_features = self.mlp(inter_features)
        bias = torch.sum(acts_topk, dim=-1, keepdim=True) if acts_topk is not None else None
        if self.attn_order == "sae_first":
            # SAE-guided attention (acts_topk/batch required)
            x = self.sae(x, acts_topk, batch)
            # Domain-guided attention (inter_features/bias required)
            for attn in self.domain:
                out = attn(x, inter_features, batch, bias)
                if isinstance(out, tuple):
                    x, ana = out
                else:
                    x, ana = out, {}
        
        elif self.attn_order == "domain_first":
            # Domain-guided attention (inter_features/bias required)
            for attn in self.domain:
                out = attn(x, inter_features, batch, bias)
                if isinstance(out, tuple):
                    x, ana = out
                else:
                    x, ana = out, {}
            # SAE-guided attention (acts_topk/batch required)
            x = self.sae(x, acts_topk, batch)
        # Residual connection
        x = x + x_i
        return x, ana

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha_dict=None, active_task_names=None, 
                 default_alpha=(1.0, 1.0)):
        """
        :param gamma: focusing parameter
        :param alpha_dict: dict mapping task name -> class weight
        :param active_task_names: list of active task names (aligned with output)
        :param default_alpha: default alpha value for missing tasks
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha_dict = alpha_dict or {}
        self.active_task_names = active_task_names or []
        self.default_alpha = default_alpha
        self.alpha_pos_vector, self.alpha_neg_vector = self._get_alpha_vector()

    def _get_alpha_vector(self):
        alpha_pos = []
        alpha_neg = []
        for task in self.active_task_names:
            alpha = self.alpha_dict.get(task, self.default_alpha)
            alpha_pos.append(alpha[0])
            alpha_neg.append(alpha[1])
        return torch.tensor(alpha_pos), torch.tensor(alpha_neg)
        # return torch.tensor([
        #     self.alpha_dict.get(task, self.default_alpha) for task in self.active_task_names
        # ])

    def forward(self, logits, targets):
        """
        logits: (B, C), raw model outputs
        targets: (B, C), 0 or 1 for each label
        """
        targets = targets.float()
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B, C)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        alpha_pos = self.alpha_pos_vector.to(logits.device)  # (C,)
        alpha_neg = self.alpha_neg_vector.to(logits.device)  # (C,)
        
        # Select the alpha for positive or negative samples
        alpha = targets * alpha_pos + (1 - targets) * alpha_neg  # (B, C)

        focal = focal * alpha  # broadcast over batch
        focal_loss = focal.mean()

        return focal_loss


class FocalSoftF1Loss(nn.Module):
    def __init__(self, gamma=2.0, alpha_dict=None, active_task_names=None,
                 default_alpha=(1.0, 1.0), softf1_weight=0.3):
        """
        :param gamma: focal loss focusing parameter
        :param alpha_dict: dict mapping task name -> class weight
        :param active_task_names: list of active task names (aligned with output)
        :param default_alpha: default alpha value for missing tasks
        :param softf1_weight: how much weight to assign to SoftF1 loss (0~1)
        """
        super(FocalSoftF1Loss, self).__init__()
        self.gamma = gamma
        self.alpha_dict = alpha_dict or {}
        self.active_task_names = active_task_names or []
        self.default_alpha = default_alpha
        self.softf1_weight = softf1_weight
        self.alpha_pos_vector, self.alpha_neg_vector = self._get_alpha_vector()

    def _get_alpha_vector(self):
        alpha_pos = []
        alpha_neg = []
        for task in self.active_task_names:
            alpha = self.alpha_dict.get(task, self.default_alpha)
            alpha_pos.append(alpha[0])
            alpha_neg.append(alpha[1])
        return torch.tensor(alpha_pos), torch.tensor(alpha_neg)
        # return torch.tensor([
        #     self.alpha_dict.get(task, self.default_alpha) for task in self.active_task_names
        # ])

    def forward(self, logits, targets):
        """
        logits: (B, C), raw model outputs
        targets: (B, C), 0 or 1 for each label
        """
        targets = targets.float()
        
        # --------- Focal Loss -------------
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B, C)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        alpha_pos = self.alpha_pos_vector.to(logits.device)  # (C,)
        alpha_neg = self.alpha_neg_vector.to(logits.device)  # (C,)
        
        # Select the alpha for positive or negative samples
        alpha = targets * alpha_pos + (1 - targets) * alpha_neg  # (B, C)

        focal = focal * alpha  # broadcast over batch
        focal_loss = focal.mean()

        # --------- SoftF1 Loss ------------
        probs = torch.sigmoid(logits)  # (B, C)
        tp = (probs * targets).sum(dim=0)  # (C,)
        fp = (probs * (1 - targets)).sum(dim=0)  # (C,)
        fn = ((1 - probs) * targets).sum(dim=0)  # (C,)

        soft_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)  # (C,)
        soft_f1_loss = 1 - soft_f1
        soft_f1_loss = soft_f1_loss.mean()

        # --------- Combined Loss ----------
        total_loss = (1 - self.softf1_weight) * focal_loss + self.softf1_weight * soft_f1_loss
        return total_loss


class IdentityGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # If the number of dimensions remains unchanged, use `nn.Identity`; otherwise, use a linear layer.
        if in_channels == out_channels:
            self.layer = nn.Identity()
        else:
            self.layer = nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_index):
        return self.layer(x)


class G4SAINT(L.LightningModule):
    def __init__(self, params):
                #  n_features, in_channels, task_names, hidden_channels, 
                #  num_layers=2, head=4,
                #  lr=1e-4, stage_epochs=(5, 5), **kwargs
        super().__init__()
        self.save_hyperparameters(params)
        self.params = params
        assert set(params['task_names']).issubset(set(index_mapping.keys())), "task_names contains invalid tasks!"
        # self.stage_epochs = stage_epochs
        self.current_stage = 1
        self.task_names = params['task_names']
        self.active_task_names = params.get("active_task_names", self.task_names) 
        # ['RNA_G4_Binding', 'all_RNA_G4_Binding',]
        self.target_indices = [index_mapping[task] for task in self.active_task_names]
        self.alpha_dict = params.get('alpha_dict', None)

        self.interpro_data = params.get('interpro_data', None)
        if self.interpro_data is None:
            self.interpro_data = InterProData(
                uniprot_ids_file="data/uniprotkb_ids.txt",
                interpro_output_file="data/uniprotkb/interproscan_pfam_output.tsv",
            )

        self.embedding_layer = nn.EmbeddingBag(len(self.interpro_data.idx_to_ipr), params['hidden_channels'], 
                                               mode='sum', include_last_offset=True)
        attn_order = params.get("attn_order", "sae_first")
        self.gnn = GNN(
            in_channels=params['in_channels'],
            hidden_channels=params['hidden_channels'],
            num_layers=params['num_layers'],
            head=params['head'],
            attn_layers=params.get("attn_layers", 2),
            use_domain=params["use_domain"],
            use_sae=params["use_sae"],
            attn_order=attn_order,
            return_analysis=params.get("return_analysis", False)
        )
        if params.get("use_identity_gcn", False):
            self.gnn.gcn = IdentityGCN(
                params['in_channels'],
                params['hidden_channels']
            )
        self.fc = Classify(params['in_channels'], params['hidden_channels'], 
                           params['task_names'], return_moe=params.get("return_moe", False))
        self.loss = FocalLoss(alpha_dict=self.alpha_dict, active_task_names=self.active_task_names)
        
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

        self.test_logits = None
        self.test_targets = None

        # self._freeze_tasks()
        # self._freeze_shared_layers()

    
    def forward(self, data):
        edge_index, batch = data.edge_index, data.batch
        acts_topk = getattr(data, 'acts_topk', None)
        x, x_pool, inter_features = self.data_process(data.x, data.prot_idx, batch)
        inter_features = self.embedding_layer(*inter_features)
        readout, ana = self.gnn(x, edge_index, inter_features, batch, acts_topk)
        readout = global_add_pool(readout, batch)
        readout = torch.cat([readout, x_pool], dim=-1)
        return self.fc(readout)
    
    def data_process(self, x, prot_idx, batch):
        with torch.no_grad():
            x = x[:, :self.params['in_channels']]
            if isinstance(prot_idx, str) or not isinstance(prot_idx, Iterable):
                prot_idx = [prot_idx]
            sample_idx = [self.interpro_data.uniprot_to_idx[pid] for pid in prot_idx]
            inter_features = self.interpro_data.get_interpro_features_by_indices(sample_idx)
            inter_features = tuple(feat.to(x.device) for feat in inter_features)
            # g = g
            # g = g[:, selected_indices]
            x_pool = global_mean_pool(x, batch)
        return x, x_pool, inter_features


    def compute_and_log_metrics(self, logits, targets, prefix, logger=False):
        targets = targets.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()
        result_fmax, _ = fmax(targets, logits)
        result_aupr = aupr(targets, logits) 
        
        # 记录 Fmax 和 AUPR
        self.log(f'{prefix}_fmax', result_fmax)
        self.log(f'{prefix}_aupr', result_aupr)
        
        if logger and self.logger is not None:
            tb_logger = self.logger.experiment
            tb_logger.add_scalars(f'{prefix}/metrics', {'Fmax': result_fmax, 'AUPR': result_aupr}, global_step=self.current_epoch)


    def __step(self, data, prefix, batch_idx):
        logits_dict = self(data)
        logits = torch.cat([logits_dict[task] for task in self.active_task_names], dim=1)
        targets = data.y[:, self.target_indices]
        loss = self.loss(logits, targets)
        self.log(f'{prefix}_loss', loss, batch_size=data.y.size(0), on_step=False, on_epoch=True)
        step_outputs = getattr(self, f'{prefix}_step_outputs')
        step_outputs.append({
            'logits': logits.detach(),
            'target': targets.detach(),
            'id': data.prot_idx
        })
        # step_outputs.append({'logits': logits.detach(), 'target': targets.detach()})
        return loss
    
    def on_after_backward(self):
        torch.cuda.empty_cache() 

    def _freeze_tasks(self):
        inactive_task_names = list(set(self.task_names) - set(self.active_task_names))
        for task in inactive_task_names:
            for param in self.fc.classifiers[task].parameters():
                param.requires_grad = False

    def _freeze_shared_layers(self):
        for module in [self.gnn, self.fc.shared_layers]:
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    def _unfreeze_shared_layers(self):
        for param in self.gnn.parameters():
            param.requires_grad = True
        for param in self.fc.shared_layers.parameters():
            param.requires_grad = True
    
    def _unfreeze_all_tasks(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def __epoch_end(self, prefix, logger=False):
        
        step_outputs = getattr(self, f'{prefix}_step_outputs')
        logits = torch.cat([x['logits'] for x in step_outputs], dim=0)  # Combine all logits
        targets = torch.cat([x['target'] for x in step_outputs], dim=0)  # Combine all targets
        ids = sum([x['id'] for x in step_outputs], [])
        step_outputs.clear()  
        
        self.compute_and_log_metrics(logits, targets, prefix, logger=logger)
        
        # Optional: Log losses to TensorBoard
        if prefix in ['train', 'val'] and self.logger:
            tb_logger = self.logger.experiment
            current_epoch = self.current_epoch
            loss = self.trainer.logged_metrics.get(f"{prefix}_loss", None)
            if loss is not None:
                tb_logger.add_scalars(
                    "Loss", {prefix: loss}, global_step=current_epoch
                )

        return logits, targets, ids
    
    def save_outputs(self, logits, targets, ids):
        if logits.ndim == 1:
            df = pd.DataFrame({
                'id': ids,
                'logit': logits,
                'target': targets
            })
        else:
            df = pd.DataFrame({
                'id': ids,
                **{f'logit_{task}': logits[:, i] for i, task in enumerate(self.active_task_names)},
                **{f'target_{task}': targets[:, i] for i, task in enumerate(self.active_task_names)}
            })
        file_name = "outputs_g4saint.csv"
        # file_name = "outputs_gnn.csv"
        df.to_csv(f"code/{file_name}", index=False)
        # np.savez("code/test_outputs.npz", logits=logits, targets=targets, ids=np.array(ids))
        # np.save("code/logits.npy", logits)
        # np.save("code/targets.npy", targets)
            

    def training_step(self, data, batch_idx):
        return self.__step(data, 'train', batch_idx)

    def on_train_epoch_end(self):
        self.__epoch_end('train')

    def validation_step(self, data, batch_idx):
        with torch.no_grad(): 
            loss = self.__step(data, 'val', batch_idx)
        return loss

    def on_validation_epoch_end(self):
        logits, targets, ids = self.__epoch_end('val')
        logits = logits.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        class_name = "all_RNA_G4_Binding"
        if class_name not in self.active_task_names:
            raise ValueError(f"{class_name} not in active_task_names")
        class_index = self.active_task_names.index(class_name)

        class_logits = logits[:, class_index]
        class_targets = targets[:, class_index]
        ap = average_precision_score(class_targets, class_logits) 
        auc_roc = roc_auc_score(class_targets, class_logits)

        self.log(f'val_ap_{class_name}', ap)
        self.log(f'val_auc_roc_{class_name}', auc_roc)

    def test_step(self, data, batch_idx):
        with torch.no_grad(): 
            loss = self.__step(data, 'test', batch_idx)
        return loss

    def on_test_epoch_end(self):
        logits, targets, ids = self.__epoch_end('test')# , logger=True

        logits = logits.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        self.test_logits = logits
        self.test_targets = targets
        prefix = 'test'

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].plot([0, 1], [0.5, 0.5], 'k--', label='Random Guess (PR)')  # PR Random Guess Reference Line
        for i, task in enumerate(self.active_task_names):
            class_idx = task
            class_logits = logits[:, i]
            class_targets = targets[:, i]

            # Calculate the Precision-Recall curve
            precisions, recalls, thresholds = precision_recall_curve(class_targets, class_logits)
            ap = average_precision_score(class_targets, class_logits)  # PR AUC

            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  
            optimal_threshold = thresholds[f1_scores.argmax()]
            class_preds = (class_logits >= optimal_threshold).astype(int)

            # Calculate various metrics
            f1 = f1_score(class_targets, class_preds, zero_division=0)
            accuracy = accuracy_score(class_targets, class_preds)

            # Plot the Precision-Recall curve for each class
            axes[0].plot(recalls, precisions, label=f'Class {class_idx} (AP = {ap:.2f})')

            self.log(f'{prefix}_ap_{class_idx}', ap)
            self.log(f'{prefix}_f1_{class_idx}', f1)
            # self.log(f'{prefix}_precision_{class_idx}', precision)
            # self.log(f'{prefix}_recall_{class_idx}', recall)
            # self.log(f'{prefix}_accuracy_{class_idx}', accuracy)

        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].legend(loc='lower left')
        axes[0].grid(alpha=0.3)


        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')  # ROC Random Guess Reference Line
        for i, task in enumerate(self.active_task_names):
            class_idx = task
            class_logits = logits[:, i]
            class_targets = targets[:, i]

            fpr, tpr, roc_thresholds = roc_curve(class_targets, class_logits)
            auc_roc = roc_auc_score(class_targets, class_logits)
            # Plot the ROC curve
            axes[1].plot(fpr, tpr, label=f'Class {class_idx} (AUC = {auc_roc:.2f})')

            # Log metrics for each class
            self.log(f'{prefix}_auc_roc_{class_idx}', auc_roc)

        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc='lower right')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        if self.params.get('save_outputs', False):
            self.save_outputs(logits, targets, ids)
            print("Testset outputs have been saved.")
        if self.params.get('save_fig', False):
            fig.savefig("code/roc_pr_curve.pdf", dpi=300, bbox_inches='tight')
            print("ROC and PR curves saved to code/roc_pr_curve.pdf")

    
    def configure_optimizers(self):
        params = self.params
        lr = params.get("lr", 1e-3)
        weight_decay = params.get("weight_decay", 0.0)
        optimizer_name = params.get("optimizer", "AdamW")
        scheduler_name = params.get("scheduler", "ReduceLROnPlateau")
        
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Configure learning rate scheduler
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=params.get("scheduler_factor", 0.5) or 0.5,
                patience=params.get("scheduler_patience", 2) or 2
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }

        elif scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        else:
            return {"optimizer": optimizer}

if __name__ == "__main__":

    g = torch.randn(16, 887)
    print(selected_indices.shape)
    g = g[:, selected_indices]
    print(g.shape)
