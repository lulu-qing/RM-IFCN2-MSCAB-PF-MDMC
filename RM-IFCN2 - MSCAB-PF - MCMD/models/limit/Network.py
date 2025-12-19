import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np

from models.fcn import FeatureEncoder 

# ==========================================
# 辅助函数
# ==========================================
def sample_task_ids(support_label, num_task, num_shot, num_way, num_class):
    basis_matrix = torch.arange(num_shot).long().view(-1, 1).repeat(1, num_way).view(-1) * num_class
    permuted_ids = torch.zeros(num_task, num_shot * num_way).long()
    permuted_labels = []
    for i in range(num_task):
        clsmap = torch.randperm(num_class)[:num_way]
        permuted_labels.append(support_label[clsmap])
        permuted_ids[i, :].copy_(basis_matrix + clsmap.repeat(num_shot))
    return permuted_ids, permuted_labels

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
    def forward(self, input_data):
        return input_data

# ==========================================
# [回退版] 多分支差异性元校准模块 (效果最佳版)
# ==========================================
class MultiBranchMDMC(nn.Module):
    def __init__(self, feat_dim=128, num_tasks=10, hidden_dim=64):
        super(MultiBranchMDMC, self).__init__()
        self.feat_dim = feat_dim
        
        # --- 1. 稳定性分支 ---
        self.stab_attn = nn.MultiheadAttention(feat_dim, num_heads=4, batch_first=True)
        self.stab_gate_net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )

        # --- 2. 可塑性分支 ---
        self.plas_diff_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        self.plas_transfer_attn = nn.MultiheadAttention(feat_dim, num_heads=4, batch_first=True)
        self.plas_proj = nn.Linear(feat_dim, feat_dim)

        # --- 3. 适应性分支 ---
        self.adapt_attn = nn.MultiheadAttention(feat_dim, num_heads=4, batch_first=True)
        self.adapt_gate_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid()
        )

        # --- 4. 三重融合 ---
        self.stage_embedding = nn.Embedding(num_tasks + 1, feat_dim)
        # 回退到单路融合缩放，这被证明是最稳健的
        self.fusion_scale = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Tanh()
        )

    def forward(self, w_old, p_new, query, task_id):
        if w_old.dim() == 2: w_old = w_old.unsqueeze(0)
        if p_new.dim() == 2: p_new = p_new.unsqueeze(0)
        if query.dim() == 2: query = query.unsqueeze(0)

        if w_old.size(1) == 0:
            has_old = False
            w_old_safe = torch.zeros_like(p_new) 
        else:
            has_old = True
            w_old_safe = w_old

        if p_new.size(1) == 0:
            return w_old.squeeze(0), p_new.squeeze(0), query.squeeze(0)

        # ==================== 1. 稳定性分支 ====================
        if has_old:
            attn_out, _ = self.stab_attn(w_old, p_new, p_new)
            mean_old = w_old.mean(dim=1)
            mean_new = p_new.mean(dim=1)
            gate_in = torch.cat([mean_old, mean_new], dim=1)
            # 保持保守约束: max 0.1
            alpha = self.stab_gate_net(gate_in).unsqueeze(1) * 0.1 
            w_old_cal = w_old + alpha * attn_out
        else:
            w_old_cal = w_old

        # ==================== 2. 可塑性分支 ====================
        N_new = p_new.size(1)
        if N_new >= 2:
            p_mean = p_new.mean(dim=1, keepdim=True)
            diff = p_new - p_mean
            enhance_vec = self.plas_diff_mlp(diff)
            p_new_enh = p_new + enhance_vec
        else:
            p_new_enh = p_new 

        if has_old:
            transfer_out, _ = self.plas_transfer_attn(p_new_enh, w_old_safe, w_old_safe)
            p_new_cal = p_new_enh + self.plas_proj(transfer_out)
        else:
            p_new_cal = p_new_enh

        # ==================== 3. 适应性分支 ====================
        memory_bank = torch.cat([w_old_cal, p_new_cal], dim=1)
        ctx_feats, _ = self.adapt_attn(query, memory_bank, memory_bank)
        beta = self.adapt_gate_net(query)
        query_cal = query + beta * ctx_feats

        # ==================== 4. 三重融合 ====================
        # 强制 reshape 为 [1, 1, D]
        gamma = self.stage_embedding(task_id).view(1, 1, -1) 
        
        # 动态约束逻辑 (Session 0 -> 0.02, Session 3 -> ~0.11)
        # 这就是之前效果最好的那个设置
        curr_sess = task_id.float().mean().item()
        dynamic_limit = 0.02 + 0.03 * curr_sess 
        dynamic_limit = min(dynamic_limit, 0.15)
        
        scale = 1 + dynamic_limit * self.fusion_scale(gamma)
        
        w_old_final = w_old_cal * scale
        p_new_final = p_new_cal * scale
        query_final = query_cal * scale

        return w_old_final.squeeze(0), p_new_final.squeeze(0), query_final.squeeze(0)

# ==========================================
# MYNET 主网络
# ==========================================
class MYNET(Net):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)
        
        if args.dataset == 'mbhm':
            self.encoder = FeatureEncoder()
            self.encoder.set_use_pf(True)
            self.num_features = 128
        else:
            self.encoder = CNNModel() 
            self.num_features = 512

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.mdmc = MultiBranchMDMC(feat_dim=self.num_features, num_tasks=args.sessions, hidden_dim=64)
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.anchor_initialized = False 

    def encode(self, x):
        x = self.encoder(x)
        if x.dim() == 3:
            x = self.avgpool(x).squeeze(-1)
        return x

    def split_instances(self, support_label, epoch):
        args = self.args
        self.current_way = args.sample_class
        permuted_ids, permuted_labels = sample_task_ids(support_label, args.num_tasks, num_shot=args.sample_shot, num_way=self.current_way, num_class=args.sample_class)
        index_label = (permuted_ids.view(args.num_tasks, args.sample_shot, self.current_way), torch.stack(permuted_labels))
        return index_label

    def forward(self, x_shot, x_query=None, shot_label=None, session=0):
        if self.mode == 'encoder':
            return self.encode(x_shot)
        else:
            support_emb = self.encode(x_shot)
            query_emb = self.encode(x_query)
            index_label = self.split_instances(shot_label, session)
            logits = self._forward(support_emb, query_emb, index_label, session)
            return logits

    def _forward(self, support, query, index_label, session):
        support_idx, support_labels = index_label
        num_task = support_idx.shape[0]
        support = support[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
        proto = support.mean(dim=1)
        
        logit_list = []
        
        if session == 'pretrain':
            for tt in range(num_task):
                current_indices = support_labels[tt, :]
                temp_weight = self.fc.weight.clone()
                temp_weight[current_indices] = proto[tt]
                query_norm = F.normalize(query, p=2, dim=1)
                cls_norm = F.normalize(temp_weight, p=2, dim=1)
                logits = torch.mm(query_norm, cls_norm.t()) * self.args.temperature
                logit_list.append(logits)
        else:
            max_sess = self.args.sessions - 1 if self.args.sessions > 1 else 1
            sim_sess = np.random.randint(0, max_sess + 1)
            task_id = torch.tensor([sim_sess]).to(support.device)

            for tt in range(num_task):
                current_indices = support_labels[tt, :]
                all_indices = torch.arange(self.args.num_classes).to(support.device)
                old_indices = all_indices[~torch.isin(all_indices, current_indices)]
                
                w_old_sim = self.fc.weight[old_indices].clone().detach()
                p_new_sim = proto[tt]
                
                w_old_cal, p_new_cal, query_cal = self.mdmc(w_old_sim, p_new_sim, query, task_id)
                
                calibrated_fc = torch.zeros_like(self.fc.weight)
                calibrated_fc[old_indices] = w_old_cal
                calibrated_fc[current_indices] = p_new_cal
                
                query_norm = F.normalize(query_cal, p=2, dim=1)
                cls_norm = F.normalize(calibrated_fc, p=2, dim=1)
                logits = torch.mm(query_norm, cls_norm.t()) * self.args.temperature
                logit_list.append(logits)
            
        logit = torch.stack(logit_list, dim=0) 
        logit = logit.view(-1, self.args.num_classes)
        return logit

    def forward_many(self, query, session=0):
        # =========================================================
        # 核心逻辑：只回退了 Logit Bias，保留了 P_new 过滤逻辑
        # =========================================================
        w_all = self.fc.weight
        
        if self.args.dataset == 'mbhm':
            base_idxs = [0, 1, 4, 7]
            inc_sessions = [[2, 3], [5, 6], [8, 9]]
        else:
            base_idxs = list(range(self.args.base_class))
            inc_sessions = [] 
            
        # 严格过滤 P_new
        active_inc_idxs = []
        if self.args.dataset == 'mbhm':
            for i in range(session):
                if i < len(inc_sessions):
                    active_inc_idxs.extend(inc_sessions[i])
        else:
            total_seen = self.args.base_class + session * self.args.way
            all_seen = list(range(total_seen))
            active_inc_idxs = [i for i in all_seen if i not in base_idxs]

        w_old = w_all[base_idxs]
        
        if len(active_inc_idxs) > 0:
            p_new = w_all[active_inc_idxs]
        else:
            p_new = torch.tensor([]).to(w_all.device).view(0, w_all.size(1))

        task_id = torch.tensor([session]).to(query.device)
        
        if self.anchor_initialized:
            w_old_cal, p_new_cal, query_cal = self.mdmc(w_old, p_new, query, task_id)
            
            refined_weights = w_all.clone()
            refined_weights[base_idxs] = w_old_cal
            if len(active_inc_idxs) > 0:
                refined_weights[active_inc_idxs] = p_new_cal
            
            final_query = query_cal
        else:
            refined_weights = w_all
            final_query = query
            
        query_norm = F.normalize(final_query, p=2, dim=1) 
        w_norm = F.normalize(refined_weights, p=2, dim=1)
        logits = F.linear(query_norm, w_norm) * self.args.temperature
        return logits

    def update_fc(self, dataloader, class_list, session):
        if isinstance(class_list, np.ndarray):
            class_list = class_list.tolist()
            
        self.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for batch in dataloader:
                data, label = [_.cuda() for _ in batch]
                embedding = self.encode(data)
                embedding_list.append(embedding)
                label_list.append(label)
        
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        for class_index in class_list:
            idx = int(class_index)
            data_index = (label_list == idx).nonzero()
            if data_index.numel() > 0:
                embedding_this = embedding_list[data_index.squeeze(-1)]
                proto_raw = embedding_this.mean(0)
                self.fc.weight.data[idx] = F.normalize(proto_raw, p=2, dim=0)