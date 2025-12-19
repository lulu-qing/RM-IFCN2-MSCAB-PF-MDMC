# models/fcn.py
# FCN with stacked MSCAB blocks and MSCAB-PF (Prototype Fusion)
# Provides interfaces to switch final block between standard MSCAB (for base-task training)
# and MSCAB_PF (for incremental prototype computation), plus helper to compute and map prototypes.
import os
import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, max(in_channels // reduction, 1), 1),
            nn.ReLU(),
            nn.Conv1d(max(in_channels // reduction, 1), in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ConvWide(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=8):
        super(ConvWide, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        # ChannelAttention kept for extension
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvMultiScale, self).__init__()
        if out_channels % 4 != 0:
            raise ValueError('out_channels should be divisible by 4')
        out_quarter = out_channels // 4
        # conv1 is residual branch with stride 4
        self.conv1 = nn.Conv1d(in_channels, out_quarter, 1, 4, padding=0)
        # multi-scale convs with stride 4
        self.conv3 = nn.Conv1d(in_channels, out_quarter, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_quarter, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_quarter, 7, 4, padding=3)
        self.norm = nn.BatchNorm1d(out_quarter * 3)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(out_quarter * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.norm(x)
        x = self.relu(x)
        x = self.ca(x) * x
        x = torch.cat([x1, x], dim=1)
        return x


class MSCAB(nn.Module):
    """
    Standard MSCAB block (multi-scale convs + channel attention + residual conv1)
    Returns dict: 'high','mid','low','residual','mult_concat','out'
    """
    def __init__(self, in_channels, out_channels, per_scale_channels=32):
        super(MSCAB, self).__init__()
        if per_scale_channels * 4 != out_channels:
            raise ValueError('out_channels must equal per_scale_channels*4')
        self.per_scale = per_scale_channels
        self.conv3 = nn.Conv1d(in_channels, per_scale_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, per_scale_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, per_scale_channels, kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(in_channels, per_scale_channels, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm1d(per_scale_channels * 3)
        self.relu = nn.ReLU()
        self.cam = ChannelAttention(per_scale_channels * 3)

    def forward(self, x):
        # x: (B, in_channels, L)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        f1 = self.conv1(x)

        mult = torch.cat([f3, f5, f7], dim=1)  # (B, 3*per_scale, L)
        mult = self.bn(mult)
        mult = self.relu(mult)

        att = self.cam(mult)  # (B, 3*per_scale, 1)
        mult_att = mult * att
        out = torch.cat([f1, mult_att], dim=1)  # (B, out_channels, L)
        return {
            'high': f3,
            'mid': f5,
            'low': f7,
            'residual': f1,
            'mult_concat': mult_att,
            'out': out
        }


class MSCAB_PF(nn.Module):
    """
    MSCAB with Prototype Fusion interfaces:
    - forward returns same dict as MSCAB
    - compute_prototypes(support_feats, labels) -> per-scale prototypes
    - gate_and_fuse(prototypes_per_scale) -> fused prototypes (num_classes, out_channels)
    """
    def __init__(self, in_channels, out_channels, per_scale_channels=32):
        super(MSCAB_PF, self).__init__()
        if per_scale_channels * 4 != out_channels:
            raise ValueError('out_channels must equal per_scale_channels*4 (i.e., 128 for per_scale=32)')
        self.per_scale = per_scale_channels
        self.conv3 = nn.Conv1d(in_channels, per_scale_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, per_scale_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, per_scale_channels, kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(in_channels, per_scale_channels, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm1d(per_scale_channels * 3)
        self.relu = nn.ReLU()
        self.cam = ChannelAttention(per_scale_channels * 3)

        self.gate_net = nn.Sequential(
            nn.Linear(per_scale_channels * 3, max(per_scale_channels * 3 // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(per_scale_channels * 3 // 2, 8), 3)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        f1 = self.conv1(x)

        mult = torch.cat([f3, f5, f7], dim=1)  # (B, 3*per_scale, L)
        mult = self.bn(mult)
        mult = self.relu(mult)

        att = self.cam(mult)
        mult_att = mult * att
        out = torch.cat([f1, mult_att], dim=1)  # (B, out_channels, L)
        return {
            'high': f3,
            'mid': f5,
            'low': f7,
            'residual': f1,
            'mult_concat': mult_att,
            'out': out
        }

    def compute_prototypes(self, support_feats: dict, labels: torch.Tensor):
        """
        support_feats: dict with keys 'high','mid','low','residual' each tensor (N_support, C_scale, L)
        labels: (N_support,)
        returns dict of per-scale prototypes with shapes (num_classes, per_scale)
        """
        device = labels.device
        s_high = self.global_pool(support_feats['high']).squeeze(-1)  # (N, per_scale)
        s_mid = self.global_pool(support_feats['mid']).squeeze(-1)
        s_low = self.global_pool(support_feats['low']).squeeze(-1)
        s_res = self.global_pool(support_feats['residual']).squeeze(-1)

        unique_labels = torch.unique(labels, sorted=True)
        prototypes_high = []
        prototypes_mid = []
        prototypes_low = []
        prototypes_res = []
        classes = []
        for c in unique_labels:
            mask = labels == c
            # if no samples for a class, create zeros (shouldn't happen if sampling correctly)
            prototypes_high.append(s_high[mask].mean(dim=0))
            prototypes_mid.append(s_mid[mask].mean(dim=0))
            prototypes_low.append(s_low[mask].mean(dim=0))
            prototypes_res.append(s_res[mask].mean(dim=0))
            classes.append(int(c.item()))

        prototypes_high = torch.stack(prototypes_high, dim=0)
        prototypes_mid = torch.stack(prototypes_mid, dim=0)
        prototypes_low = torch.stack(prototypes_low, dim=0)
        prototypes_res = torch.stack(prototypes_res, dim=0)
        return {
            'high': prototypes_high,
            'mid': prototypes_mid,
            'low': prototypes_low,
            'residual': prototypes_res,
            'classes': torch.tensor(classes, dtype=torch.long)
        }

    def gate_and_fuse(self, prototypes_per_scale: dict):
        """
        prototypes_per_scale: dict from compute_prototypes
        returns: fused prototypes tensor (num_classes, out_channels) and gate weights (num_classes, 3)
        """
        P_high = prototypes_per_scale['high']
        P_mid = prototypes_per_scale['mid']
        P_low = prototypes_per_scale['low']
        P_res = prototypes_per_scale['residual']
        Cnum = P_high.size(0)

        gate_in = torch.cat([P_high, P_mid, P_low], dim=1)  # (Cnum, 3*per_scale)
        gate_logits = self.gate_net(gate_in)  # (Cnum, 3)
        gate_w = F.softmax(gate_logits, dim=1)

        a_h = gate_w[:, 0].unsqueeze(-1)
        a_m = gate_w[:, 1].unsqueeze(-1)
        a_l = gate_w[:, 2].unsqueeze(-1)

        fused_scales = torch.cat([a_h * P_high, a_m * P_mid, a_l * P_low], dim=1)  # (Cnum, 3*per_scale)
        fused = torch.cat([P_res, fused_scales], dim=1)  # (Cnum, out_channels)
        return fused, gate_w


class FeatureEncoder(nn.Module):
    """
    Feature encoder composed as:
      - ConvWide branches (query/ref/res)
      - ConvMultiScale + MSCAB (stage1)
      - ConvMultiScale + MSCAB (stage2)
      - ConvMultiScale + MSCAB_OR_PF (stage3)  <-- switchable (use_pf)
    Provide convenience methods:
      - set_use_pf(flag)
      - forward_to_penultimate(x) -> features before final MSCAB
      - get_multiscale_outputs(x, use_pf=True) -> dict from final block
    """
    def __init__(self, per_scale_channels=32):
        super(FeatureEncoder, self).__init__()
        self.conv_query = ConvWide(1, 60, 8, 8)
        self.conv_ref = ConvWide(1, 8, 8, 8)
        self.conv_res = ConvWide(1, 60, 8, 8)

        # stage1
        self.conv1 = ConvMultiScale(128, 128)
        self.mscab1 = MSCAB(128, 128, per_scale_channels=per_scale_channels)
        # stage2
        self.conv2 = ConvMultiScale(128, 128)
        self.mscab2 = MSCAB(128, 128, per_scale_channels=per_scale_channels)
        # stage3 (final)
        self.conv3 = ConvMultiScale(128, 128)
        self.mscab3 = MSCAB(128, 128, per_scale_channels=per_scale_channels)      # standard final MSCAB
        self.mscab_pf = MSCAB_PF(128, 128, per_scale_channels=per_scale_channels)  # PF final MSCAB

        # control switch: if True use PF final block, else use standard final block
        self.use_pf = False

    def set_use_pf(self, flag: bool):
        self.use_pf = bool(flag)

    def forward_to_penultimate(self, x):
        """
        Run encoder up to the input of final MSCAB (i.e., return features before final block).
        Useful for operations that need to modify last-block input or weights prior to last block.
        """
        if x.dim() == 3 and x.size(1) == 1:
            x = x.repeat(1, 3, 1)
        query = x[:, :1, :]
        ref = x[:, 1:2, :]
        res = query - ref

        query = self.conv_query(query)
        ref = self.conv_ref(ref)
        res = self.conv_res(res)

        x = torch.cat([query, ref, res], dim=1)
        x = self.conv1(x)
        x = self.mscab1(x)['out']
        x = self.conv2(x)
        x = self.mscab2(x)['out']
        x = self.conv3(x)
        return x  # (B, 128, L_before_final)

    def forward(self, x):
        """
        Full forward. Final block is chosen by self.use_pf flag.
        """
        x_pen = self.forward_to_penultimate(x)
        if self.use_pf:
            mscab_out = self.mscab_pf(x_pen)
        else:
            mscab_out = self.mscab3(x_pen)
        return mscab_out['out']

    def get_multiscale_outputs(self, x, use_pf=None):
        """
        Returns the multiscale dict from the final block.
        If use_pf is None, use self.use_pf; otherwise override with provided flag.
        """
        if use_pf is None:
            use_pf = self.use_pf
        if x.dim() == 3 and x.size(1) == 1:
            x = x.repeat(1, 3, 1)
        query = x[:, :1, :]
        ref = x[:, 1:2, :]
        res = query - ref

        query = self.conv_query(query)
        ref = self.conv_ref(ref)
        res = self.conv_res(res)

        x = torch.cat([query, ref, res], dim=1)
        x = self.conv1(x)
        x = self.mscab1(x)['out']
        x = self.conv2(x)
        x = self.mscab2(x)['out']
        x = self.conv3(x)
        if use_pf:
            return self.mscab_pf(x)
        else:
            return self.mscab3(x)


class Classifier(nn.Module):
    """
    Classifier that pools encoder outputs over time then projects to hidden_dim -> logits.
    Also supports prototype-based classification where prototypes are in hidden_dim space.
    """
    def __init__(self, feat_channels=128, hidden_dim=128, num_classes=10):
        super(Classifier, self).__init__()
        self.feat_channels = feat_channels
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(self.feat_channels, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, num_classes)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # prototype storage (hidden_dim space)
        self.prototype_weights = None
        self.prototype_scale = 10.0
        # prototype_classes metadata (list of global ids corresponding to prototype rows)
        self.prototype_classes = None

    def forward(self, x):
        # x: (B, C, L) or (B, C)
        if x.dim() == 3:
            x = self.global_pool(x).squeeze(-1)  # (B, C)
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

    def embed(self, x):
        # return embedding in hidden space (B, hidden_dim)
        if x.dim() == 3:
            x = self.global_pool(x).squeeze(-1)
        x = self.linear1(x)
        x = torch.relu(x)
        return x

    def set_prototypes(self, prototypes: torch.Tensor):
        """
        prototypes: (num_new_classes, hidden_dim)
        """
        if prototypes.ndim != 2:
            raise ValueError("prototypes must be 2D tensor")
        if prototypes.size(1) != self.hidden_dim:
            raise ValueError(f"Prototype dim {prototypes.size(1)} doesn't match classifier hidden dim {self.hidden_dim}")
        self.prototype_weights = F.normalize(prototypes, p=2, dim=1)

    def classify_with_prototypes(self, embeddings: torch.Tensor):
        """
        embeddings: (B, hidden_dim)
        returns logits: (B, num_prototypes)
        """
        if self.prototype_weights is None:
            raise RuntimeError("No prototypes set. Call set_prototypes() first.")
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        logits = torch.matmul(emb_norm, self.prototype_weights.t()) * self.prototype_scale
        return logits


class FaultClassificationNetwork(nn.Module):
    """
    High-level network that exposes:
     - encoder (FeatureEncoder)
     - classifier (Classifier)
     - methods to compute fused prototypes from support set using MSCAB_PF and map them
       into classifier hidden space (set_prototypes).
    """
    def __init__(self, num_classes=10, input_length=24000, per_scale_channels=32):
        super(FaultClassificationNetwork, self).__init__()
        self.encoder = FeatureEncoder(per_scale_channels=per_scale_channels)
        self.classifier = Classifier(feat_channels=128, hidden_dim=128, num_classes=num_classes)
        # optional projection if you want to map fused prototypes into other spaces (not mandatory)
        # self.prototype_projection = nn.Linear(128, 128)  # identity-like if dims same

    def forward(self, x):
        feat_map = self.encoder(x)  # (B, 128, L_final)
        return self.classifier(feat_map)

    def get_embeddings(self, x):
        feat_map = self.encoder(x)
        emb = self.classifier.embed(feat_map)  # (B, hidden_dim)
        return emb

    def get_multiscale_outputs(self, x, use_pf=None):
        return self.encoder.get_multiscale_outputs(x, use_pf=use_pf)

    def set_encoder_use_pf(self, flag: bool):
        self.encoder.set_use_pf(flag)

    def compute_and_set_prototypes_from_support(self, support_x, support_y, device=None, set_as_classifier_prototypes=True):
        """
        Compute fused prototypes from support set using encoder.mscab_pf and set them into classifier.

        support_x: (N_support, C, L)
        support_y: (N_support,) labels (global class ids or local ones depending on usage)
        device: target device (optional)
        Returns: fused (num_classes, feat_channels), mapped (num_classes, hidden_dim), gate_w, classes
        If set_as_classifier_prototypes=True, classifier.prototype_weights will be set to mapped prototypes.
        """
        if device is None:
            device = next(self.parameters()).device

        # Ensure final block is PF
        prev_flag = self.encoder.use_pf
        self.encoder.set_use_pf(True)

        # Move inputs to device
        support_x = support_x.to(device)
        support_y = support_y.to(device)

        # Get multiscale outputs from final PF block
        ms = self.encoder.get_multiscale_outputs(support_x, use_pf=True)
        mscab_pf = self.encoder.mscab_pf

        # compute per-scale prototypes
        prototypes_per_scale = mscab_pf.compute_prototypes({
            'high': ms['high'],
            'mid': ms['mid'],
            'low': ms['low'],
            'residual': ms['residual']
        }, support_y)

        # fuse via gate network
        fused, gate_w = mscab_pf.gate_and_fuse(prototypes_per_scale)  # fused: (num_new, feat_channels)

        # Map fused (feat space) into classifier hidden dim via classifier.linear1 (same mapping used for embeddings)
        with torch.no_grad():
            mapped = self.classifier.linear1(fused.to(device))  # (num_new, hidden_dim)
            mapped = torch.relu(mapped)

        # Optionally set as classifier prototypes (for prototype-based inference)
        if set_as_classifier_prototypes:
            try:
                self.classifier.set_prototypes(mapped)
                # record prototype class ids so we can align them at test time
                classes = prototypes_per_scale.get('classes', None)
                if classes is not None:
                    try:
                        cls_list = classes.cpu().tolist() if isinstance(classes, torch.Tensor) else list(map(int, classes))
                    except Exception:
                        cls_list = [int(x) for x in classes]
                    self.classifier.prototype_classes = cls_list
                else:
                    self.classifier.prototype_classes = None
            except Exception:
                self.classifier.prototype_classes = None

        # restore previous flag
        self.encoder.set_use_pf(prev_flag)
        return fused, mapped, gate_w, prototypes_per_scale.get('classes', None)

    def extend_classifier_with_prototypes(self, fused_prototypes):
        """
        Backwards-compatible helper.
        Accept fused_prototypes: (num_new, feat_channels) (e.g., output of gate_and_fuse),
        map to classifier hidden space and set as classifier prototypes for prototype-based inference.
        Returns mapped prototypes (num_new, hidden_dim).
        """
        device = next(self.parameters()).device
        fused = fused_prototypes.to(device)
        with torch.no_grad():
            mapped = self.classifier.linear1(fused)  # (N, hidden_dim)
            mapped = torch.relu(mapped)
            self.classifier.set_prototypes(mapped)
        return mapped

    def save_weights(self, weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(weights_dir, 'feature_encoder.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(weights_dir, 'classifier.pth'))

    def load_weights(self, weights_dir, map_location='cpu'):
        self.encoder.load_state_dict(torch.load(os.path.join(weights_dir, 'feature_encoder.pth'), map_location=map_location))
        self.classifier.load_state_dict(torch.load(os.path.join(weights_dir, 'classifier.pth'), map_location=map_location))


if __name__ == '__main__':
    # quick smoke test (CPU)
    def test_smoke():
        B = 8
        L = 24000
        x = torch.randn(B, 3, L)
        model = FaultClassificationNetwork(num_classes=10)
        # base-mode (use standard MSCAB)
        model.set_encoder_use_pf(False)
        logits = model(x)
        print("legacy logits shape:", logits.shape)
        # now test PF flow
        model.set_encoder_use_pf(True)
        ms = model.get_multiscale_outputs(x)
        print("ms keys:", list(ms.keys()))
        # small support/test
        support_x = x[:6]
        labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        fused, mapped, gate_w, classes = model.compute_and_set_prototypes_from_support(support_x, labels, device='cpu')
        print("fused shape:", fused.shape, "mapped shape:", mapped.shape, "gate_w shape:", gate_w.shape)
        # prototype-based classification test
        emb = model.get_embeddings(x[:4])
        logits_proto = model.classifier.classify_with_prototypes(emb)
        print("prototype logits shape:", logits_proto.shape)

    test_smoke()