import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.motion_magnification_learning_based_master.magnet import Manipulator as MagManipulator
from third_party.motion_magnification_learning_based_master.magnet import Encoder_No_texture as MagEncoder_No_texture
from ramoe.models.region_classifier import RegionClassifier


def gen_state_dict(weights_path: str):
    st = torch.load(weights_path, map_location="cpu")
    state_dict = {}
    for k, v in st.items():
        state_dict[k.replace("module.", "")] = v
    return state_dict


class ConsensusModule(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type if consensus_type != "rnn" else "identity"
        self.dim = dim

    def forward(self, x):
        return SegmentConsensus(self.consensus_type, self.dim)(x)


class SegmentConsensus(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super().__init__()
        self.consensus_type = consensus_type
        self.dim = dim

    def forward(self, x):
        if self.consensus_type == "avg":
            return x.mean(dim=self.dim, keepdim=True)
        if self.consensus_type == "identity":
            return x
        return None


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print("=> Using in-place shift...")
        print("=> Using fold div: {}".format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            raise NotImplementedError("inplace shift is not implemented")
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                 # shift left
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift right
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]          # not shift
        return out.view(nt, c, h, w)


class eca_layer_2d_v2(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k_size = t if t % 2 else (t + 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)

        y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_max = self.conv(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y_avg + y_max)
        return x * y.expand_as(x)


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = pred * target + (1 - pred) * (1 - target)
    loss = alpha * ((1 - pt) ** gamma) * bce
    return loss.mean()


def feature_loss_function(fea, target_fea):
    return F.mse_loss(fea, target_fea, reduction="mean")


def new_kd_loss_function(outputs, teacher_outputs, T):
    outputs = F.log_softmax(outputs / T, dim=1)
    teacher_outputs = F.softmax(teacher_outputs / T, dim=1)
    loss = F.kl_div(outputs, teacher_outputs, reduction="sum") * (T**2) / outputs.shape[0]
    return loss


class SimpleMoEFusion(nn.Module):
    """
    简洁 MoE (3 experts): S / T / AU
    - 无文本引导；train/test gating 一致
    """
    def __init__(self, class_num, d_moe, dim_S, dim_T, dim_AU, gate_hidden=128, dropout=0.0):
        super().__init__()
        self.proj_S = nn.Linear(dim_S, d_moe)
        self.proj_T = nn.Linear(dim_T, d_moe)
        self.proj_AU = nn.Linear(dim_AU, d_moe)

        self.gate = nn.Sequential(
            nn.Linear(d_moe * 3, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )
        self.head = nn.Linear(d_moe, class_num)

    def forward(self, eS, eT, eAU):
        zS = self.proj_S(eS)
        zT = self.proj_T(eT)
        zAU = self.proj_AU(eAU)

        gate_logits = self.gate(torch.cat([zS, zT, zAU], dim=-1))  # [B,3]
        gate_w = F.softmax(gate_logits, dim=1)                     # [B,3]

        Z = torch.stack([zS, zT, zAU], dim=1)                      # [B,3,d]
        fused_feat = (gate_w.unsqueeze(-1) * Z).sum(dim=1)         # [B,d]

        logits = self.head(fused_feat)
        return logits, fused_feat, gate_w


class SKD_TSTSAN(nn.Module):
    def __init__(self, out_channels=5, amp_factor=5):
        super().__init__()

        # Motion magnification (原三流逻辑保持)
        self.Aug_Encoder_L = MagEncoder_No_texture(dim_in=16)
        self.Aug_Encoder_S = MagEncoder_No_texture(dim_in=1)
        self.Aug_Encoder_T = MagEncoder_No_texture(dim_in=2)
        self.Aug_Manipulator_L = MagManipulator()
        self.Aug_Manipulator_S = MagManipulator()
        self.Aug_Manipulator_T = MagManipulator()

        # L/S/T backbone
        self.conv1_L = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv1_S = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv1_T = nn.Conv2d(32, 64, kernel_size=5, stride=1)

        self.relu = nn.ReLU()
        self.bn1_L = nn.BatchNorm2d(64)
        self.bn1_S = nn.BatchNorm2d(64)
        self.bn1_T = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)

        # AC1
        self.AC1_conv1_L = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.AC1_conv1_S = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.AC1_conv1_T = TemporalShift(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), n_segment=2, n_div=8)

        self.AC1_bn1_L = nn.BatchNorm2d(128)
        self.AC1_bn1_S = nn.BatchNorm2d(128)
        self.AC1_bn1_T = nn.BatchNorm2d(128)

        self.AC1_conv2_L = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.AC1_conv2_S = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.AC1_conv2_T = TemporalShift(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), n_segment=2, n_div=8)

        self.AC1_bn2_L = nn.BatchNorm2d(128)
        self.AC1_bn2_S = nn.BatchNorm2d(128)
        self.AC1_bn2_T = nn.BatchNorm2d(128)

        self.AC1_pool = nn.AdaptiveAvgPool2d(1)
        self.AC1_fc = nn.Linear(in_features=384, out_features=out_channels)

        # mid blocks
        self.conv2_L = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_S = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_T = TemporalShift(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), n_segment=2, n_div=8)
        self.bn2_L = nn.BatchNorm2d(64)
        self.bn2_S = nn.BatchNorm2d(64)
        self.bn2_T = nn.BatchNorm2d(64)

        self.conv3_L = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_S = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_T = TemporalShift(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), n_segment=2, n_div=8)
        self.bn3_L = nn.BatchNorm2d(64)
        self.bn3_S = nn.BatchNorm2d(64)
        self.bn3_T = nn.BatchNorm2d(64)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # AC2
        self.AC2_conv1_L = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.AC2_conv1_S = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.AC2_conv1_T = TemporalShift(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), n_segment=2, n_div=8)

        self.AC2_bn1_L = nn.BatchNorm2d(128)
        self.AC2_bn1_S = nn.BatchNorm2d(128)
        self.AC2_bn1_T = nn.BatchNorm2d(128)

        self.AC2_conv2_L = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.AC2_conv2_S = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.AC2_conv2_T = TemporalShift(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), n_segment=2, n_div=8)

        self.AC2_bn2_L = nn.BatchNorm2d(128)
        self.AC2_bn2_S = nn.BatchNorm2d(128)
        self.AC2_bn2_T = nn.BatchNorm2d(128)

        self.AC2_pool = nn.AdaptiveAvgPool2d(1)
        self.AC2_fc = nn.Linear(in_features=384, out_features=out_channels)

        # tail
        self.all_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv4_L = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_S = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_T = TemporalShift(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), n_segment=2, n_div=8)
        self.bn4_L = nn.BatchNorm2d(128)
        self.bn4_S = nn.BatchNorm2d(128)
        self.bn4_T = nn.BatchNorm2d(128)

        self.conv5_L = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5_S = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5_T = TemporalShift(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), n_segment=2, n_div=8)
        self.bn5_L = nn.BatchNorm2d(128)
        self.bn5_S = nn.BatchNorm2d(128)
        self.bn5_T = nn.BatchNorm2d(128)

        self.fc2 = nn.Linear(in_features=512, out_features=out_channels)

        # attentions
        self.ECA1 = eca_layer_2d_v2(64)
        self.ECA2 = eca_layer_2d_v2(64)
        self.ECA3 = eca_layer_2d_v2(64)
        self.ECA4 = eca_layer_2d_v2(128)
        self.ECA5 = eca_layer_2d_v2(128)

        self.AC1_ECA1 = eca_layer_2d_v2(128)
        self.AC2_ECA1 = eca_layer_2d_v2(128)

        self.amp_factor = amp_factor
        self.consensus = ConsensusModule("avg")
        self.dropout = nn.Dropout(0.2)

        # ---------------- AU branch + RegionNet gating ----------------
        self.region_net = RegionClassifier()

        self.conv1_AU = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1_AU = nn.BatchNorm2d(64)

        self.conv2_AU = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_AU = nn.BatchNorm2d(64)

        self.conv3_AU = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_AU = nn.BatchNorm2d(64)
        self.avgpool_AU = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4_AU = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_AU = nn.BatchNorm2d(128)

        self.conv5_AU = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5_AU = nn.BatchNorm2d(128)
        self.all_avgpool_AU = nn.AdaptiveAvgPool2d(1)

        # AC1 AU
        self.AC1_conv1_AU = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.AC1_bn1_AU = nn.BatchNorm2d(128)
        self.AC1_conv2_AU = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.AC1_bn2_AU = nn.BatchNorm2d(128)
        self.AC1_pool_AU = nn.AdaptiveAvgPool2d(1)

        # AC2 AU
        self.AC2_conv1_AU = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.AC2_bn1_AU = nn.BatchNorm2d(128)
        self.AC2_conv2_AU = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.AC2_bn2_AU = nn.BatchNorm2d(128)
        self.AC2_pool_AU = nn.AdaptiveAvgPool2d(1)

        # ---------------- MoE (简洁版，无文本引导；保留接口不变) ----------------
        self.moe = SimpleMoEFusion(
            class_num=out_channels,
            d_moe=384,
            dim_S=128,
            dim_T=128,
            dim_AU=128,
            gate_hidden=128,
            dropout=0.0,
        )

    def forward(self, input, label=None):
        # input: [B, C, 48, 48]  (按你数据构造的通道顺序)
        x1 = input[:, 2:18, :, :]      # L
        x1_onset = input[:, 18:34, :, :]
        x2 = input[:, 0, :, :].unsqueeze(1)   # S
        x2_onset = input[:, 1, :, :].unsqueeze(1)
        x3 = input[:, 34:38, :, :]     # T (4ch)
        x_au = input[:, 0, :, :].unsqueeze(1)  # AU 输入用灰度
        x_au0 = x_au

        # Region 5ch mask: [B,5,48,48] (你数据里是 5 个通道逐个 append 的)
        region_mask_5ch = input[:, 38:43, :, :]

        bsz = x1.shape[0]

        # T reshape: [B*2, 2, 48,48]
        x3 = torch.reshape(x3, (bsz * 2, 2, 48, 48))
        x3_onset = torch.zeros(bsz * 2, 2, 48, 48, device=input.device)

        # ---------------- Motion magnification (L/S/T) ----------------
        motion_x1_onset = self.Aug_Encoder_L(x1_onset)
        motion_x1 = self.Aug_Encoder_L(x1)
        x1 = self.Aug_Manipulator_L(motion_x1_onset, motion_x1, self.amp_factor)

        motion_x2_onset = self.Aug_Encoder_S(x2_onset)
        motion_x2 = self.Aug_Encoder_S(x2)
        x2 = self.Aug_Manipulator_S(motion_x2_onset, motion_x2, self.amp_factor)

        motion_x3_onset = self.Aug_Encoder_T(x3_onset)
        motion_x3 = self.Aug_Encoder_T(x3)
        x3 = self.Aug_Manipulator_T(motion_x3_onset, motion_x3, self.amp_factor)

        # ---------------- AU branch + RegionNet gating ----------------
        x_au = self.relu(self.bn1_AU(self.conv1_AU(x_au)))
        x_au = self.maxpool(x_au)

        region_logits = self.region_net(x_au0)             # [B,5]
        region_weight = torch.sigmoid(region_logits)       # [B,5]
        region_weight = region_weight.unsqueeze(-1).unsqueeze(-1)  # [B,5,1,1]

        weighted_region_mask = region_weight * region_mask_5ch      # [B,5,48,48]
        soft_region_mask = weighted_region_mask.sum(dim=1, keepdim=True)  # [B,1,48,48]

        soft_region_mask = F.interpolate(
            soft_region_mask, size=x_au.shape[-2:], mode="bilinear", align_corners=False
        )  # [B,1,24,24]

        x_au_masked = x_au * soft_region_mask

        # ---------------- L/S/T conv stem ----------------
        x1 = self.relu(self.bn1_L(self.conv1_L(x1)))
        x1 = self.ECA1(x1)
        x1 = self.maxpool(x1)

        x2 = self.relu(self.bn1_S(self.conv1_S(x2)))
        x2 = self.maxpool(x2)

        x3 = self.relu(self.bn1_T(self.conv1_T(x3)))
        x3 = self.maxpool(x3)

        # ---------------- AC1 (带 AU) ----------------
        AC1_x_au = self.relu(self.AC1_bn1_AU(self.AC1_conv1_AU(x_au_masked)))
        AC1_x_au = self.relu(self.AC1_bn2_AU(self.AC1_conv2_AU(AC1_x_au)))
        AC1_x_au = self.AC1_pool(AC1_x_au)
        AC1_x_au_all = AC1_x_au.view(AC1_x_au.size(0), -1)

        AC1_x1 = self.relu(self.AC1_bn1_L(self.AC1_conv1_L(x1)))
        AC1_x1 = self.AC1_ECA1(AC1_x1)
        AC1_x1 = self.relu(self.AC1_bn2_L(self.AC1_conv2_L(AC1_x1)))
        AC1_x1 = self.AC1_pool(AC1_x1)
        AC1_x1_all = AC1_x1.view(AC1_x1.size(0), -1)

        AC1_x2 = self.relu(self.AC1_bn1_S(self.AC1_conv1_S(x2)))
        AC1_x2 = self.relu(self.AC1_bn2_S(self.AC1_conv2_S(AC1_x2)))
        AC1_x2 = self.AC1_pool(AC1_x2)
        AC1_x2_all = AC1_x2.view(AC1_x2.size(0), -1)

        AC1_x3 = self.relu(self.AC1_bn1_T(self.AC1_conv1_T(x3)))
        AC1_x3 = self.relu(self.AC1_bn2_T(self.AC1_conv2_T(AC1_x3)))
        AC1_x3 = self.AC1_pool(AC1_x3)
        AC1_x3_all = AC1_x3.view(AC1_x3.size(0), -1)
        AC1_x3_all = torch.reshape(AC1_x3_all, (bsz, 2, -1))
        AC1_x3_all = self.consensus(AC1_x3_all)
        AC1_x3_all = torch.reshape(AC1_x3_all, (bsz, -1))

        AC1_all = torch.cat((AC1_x2_all, AC1_x3_all, AC1_x_au_all), dim=1)
        AC1_all = self.dropout(AC1_all)
        AC1_out = self.AC1_fc(AC1_all)

        # ---------------- mid blocks ----------------
        x1 = self.relu(self.bn2_L(self.conv2_L(x1)))
        x1 = self.ECA2(x1)
        x2 = self.relu(self.bn2_S(self.conv2_S(x2)))
        x3 = self.relu(self.bn2_T(self.conv2_T(x3)))

        x_au_masked = self.relu(self.bn2_AU(self.conv2_AU(x_au_masked)))

        x1 = self.relu(self.bn3_L(self.conv3_L(x1)))
        x1 = self.ECA3(x1)
        x1 = self.avgpool(x1)

        x2 = self.relu(self.bn3_S(self.conv3_S(x2)))
        x2 = self.avgpool(x2)

        x3 = self.relu(self.bn3_T(self.conv3_T(x3)))
        x3 = self.avgpool(x3)

        x_au_masked = self.relu(self.bn3_AU(self.conv3_AU(x_au_masked)))
        x_au_masked = self.avgpool_AU(x_au_masked)

        # ---------------- AC2 (带 AU) ----------------
        AC2_x_au = self.relu(self.AC2_bn1_AU(self.AC2_conv1_AU(x_au_masked)))
        AC2_x_au = self.relu(self.AC2_bn2_AU(self.AC2_conv2_AU(AC2_x_au)))
        AC2_x_au = self.AC2_pool(AC2_x_au)
        AC2_x_au_all = AC2_x_au.view(AC2_x_au.size(0), -1)

        AC2_x1 = self.relu(self.AC2_bn1_L(self.AC2_conv1_L(x1)))
        AC2_x1 = self.AC2_ECA1(AC2_x1)
        AC2_x1 = self.relu(self.AC2_bn2_L(self.AC2_conv2_L(AC2_x1)))
        AC2_x1 = self.AC2_pool(AC2_x1)
        AC2_x1_all = AC2_x1.view(AC2_x1.size(0), -1)

        AC2_x2 = self.relu(self.AC2_bn1_S(self.AC2_conv1_S(x2)))
        AC2_x2 = self.relu(self.AC2_bn2_S(self.AC2_conv2_S(AC2_x2)))
        AC2_x2 = self.AC2_pool(AC2_x2)
        AC2_x2_all = AC2_x2.view(AC2_x2.size(0), -1)

        AC2_x3 = self.relu(self.AC2_bn1_T(self.AC2_conv1_T(x3)))
        AC2_x3 = self.relu(self.AC2_bn2_T(self.AC2_conv2_T(AC2_x3)))
        AC2_x3 = self.AC2_pool(AC2_x3)
        AC2_x3_all = AC2_x3.view(AC2_x3.size(0), -1)
        AC2_x3_all = torch.reshape(AC2_x3_all, (bsz, 2, -1))
        AC2_x3_all = self.consensus(AC2_x3_all)
        AC2_x3_all = torch.reshape(AC2_x3_all, (bsz, -1))

        AC2_all = torch.cat(( AC2_x2_all, AC2_x3_all, AC2_x_au_all), dim=1)
        AC2_all = self.dropout(AC2_all)
        AC2_out = self.AC2_fc(AC2_all)

        # ---------------- tail ----------------
        x1 = self.relu(self.bn4_L(self.conv4_L(x1)))
        x1 = self.ECA4(x1)
        x2 = self.relu(self.bn4_S(self.conv4_S(x2)))
        x3 = self.relu(self.bn4_T(self.conv4_T(x3)))

        x_au_masked = self.relu(self.bn4_AU(self.conv4_AU(x_au_masked)))

        x1 = self.relu(self.bn5_L(self.conv5_L(x1)))
        x1 = self.ECA5(x1)
        x1 = self.all_avgpool(x1)
        x1_all = x1.view(x1.size(0), -1)

        x2 = self.relu(self.bn5_S(self.conv5_S(x2)))
        x2 = self.all_avgpool(x2)
        x2_all = x2.view(x2.size(0), -1)

        x3 = self.relu(self.bn5_T(self.conv5_T(x3)))
        x3 = self.all_avgpool(x3)
        x3_all = x3.view(x3.size(0), -1)
        x3_all = torch.reshape(x3_all, (bsz, 2, -1))
        x3_all = self.consensus(x3_all)
        x3_all = torch.reshape(x3_all, (bsz, -1))

        x_au_masked = self.relu(self.bn5_AU(self.conv5_AU(x_au_masked)))
        x_au_masked = self.all_avgpool_AU(x_au_masked)
        x_au_all = x_au_masked.view(x_au_masked.size(0), -1)

        # ---------------- MoE (S/T/AU) ----------------
        # 保持 forward 接口不变：依然接收 label，但 gating 不再使用 label（train/test 一致）
        y, final_feature, gate_w = self.moe(x2_all, x3_all, x_au_all)

        # 保持返回值不变
        return y, AC1_out, AC2_out, final_feature, AC1_all, AC2_all


def get_model(model_name, num_classes, alpha):
    if model_name == "SKD_TSTSAN":
        return SKD_TSTSAN(out_channels=num_classes, amp_factor=alpha)
    raise ValueError(f"Unknown model name: {model_name}")
