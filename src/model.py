from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from torch.nn.init import trunc_normal_
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from utils.adapter_modules import SimpleAdapter, SimpleProj
from utils.descriptions import DESCRIPTIONS_ORI, DESCRIPTIONS_ORI_XD
from utils.dnp_vision_transformer import Aggregation_Block, Prototype_Block
import math


RUPA_PARAMETER_PREFIXES = ('rupa_gate.',)

def load_compatible_model_state(model, state_dict, allow_legacy_dsanet=False):
    target_model = model.module if hasattr(model, 'module') else model
    incompatible = target_model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = (
        [key for key in missing if key.startswith(RUPA_PARAMETER_PREFIXES)]
        if allow_legacy_dsanet else []
    )
    disallowed_missing = [key for key in missing if key not in allowed_missing]
    if disallowed_missing or unexpected:
        raise RuntimeError('Incompatible checkpoint.')



class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, query_feat, key_value_feat):
        attn_output, _ = self.cross_attn(query=query_feat, key=key_value_feat, value=key_value_feat)
        x = self.ln1(query_feat + attn_output)

        ffn_output = self.ffn(x)
        out = self.ln2(x + ffn_output)
        return out

class CrossModalFusionTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query_feat, key_value_feat):
        x = query_feat
        for layer in self.layers:
            x = layer(query_feat=x, key_value_feat=key_value_feat)
        return x

class CLIP_Adapter(nn.Module):
    def __init__(self, clipmodel, device, text_adapt_until=3, t_w=0.1):
        super(CLIP_Adapter, self).__init__()
        self.clipmodel = clipmodel
        self.text_adapt_until = text_adapt_until
        self.t_w = t_w
        self.device = device

        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(512, 512) for _ in range(text_adapt_until)] +
            [SimpleProj(512, 512, relu=True)]
        )

        self._init_weights_()

    def _init_weights_(self):
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)

        cast_dtype = self.clipmodel.token_embedding.weight.dtype

        x = self.clipmodel.token_embedding(text).to(cast_dtype)

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)

        for i in range(len(self.clipmodel.transformer.resblocks)):
            x = self.clipmodel.transformer.resblocks[i](x)
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out * x.norm(dim=-1, keepdim=True) /
                    (adapt_out.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x

        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        eot_indices = text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices]
        x = self.text_adapter[-1](x)

        return x

class SGNM(nn.Module):
    def __init__(self, feature_dim=512, num_prototypes=64, num_heads=8,
                 extractor_depth=1, decoder_depth=8, normal_selection_ratio=0.125,
                 adaptive_normal_selection=True, min_normal_frames=4,
                 max_normal_ratio=0.8):
        super().__init__()

        self.normal_selection_ratio = normal_selection_ratio
        self.adaptive_normal_selection = adaptive_normal_selection
        self.min_normal_frames = min_normal_frames
        self.max_normal_ratio = max_normal_ratio
        self.video_prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.dnp_extractor = nn.ModuleList([
            Aggregation_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)
            ) for _ in range(extractor_depth)
        ])

        self.bottleneck = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(feature_dim, feature_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(feature_dim * 4, feature_dim))
        ]))

        self.decoder = nn.ModuleList([
            Prototype_Block(
                dim=feature_dim, num_heads=num_heads, mlp_ratio=4.,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)
            ) for _ in range(decoder_depth)
        ])

    @staticmethod
    def otsu_threshold_1d(scores):
        """Compute Otsu's optimal threshold on a 1-D tensor of anomaly scores.

        The algorithm finds the threshold t in [min, max] that maximises the
        inter-class variance σ²_B(t) = w0 * w1 * (μ0 − μ1)², effectively
        splitting the score distribution into *normal* (< t) and *anomaly*
        (>= t) groups.

        Args:
            scores: 1-D float tensor of anomaly scores (already valid, no inf).

        Returns:
            threshold (float): the optimal cut-off value.
        """
        sorted_scores, _ = scores.sort()
        n = sorted_scores.numel()
        if n <= 1:
            return float(sorted_scores[0].item()) + 1e-6

        # Use 256 candidate thresholds uniformly spanning the score range
        # (or fewer when the number of unique values is small).
        num_bins = min(256, n)
        s_min = sorted_scores[0].item()
        s_max = sorted_scores[-1].item()
        if s_max - s_min < 1e-8:
            # All scores are (nearly) identical — treat all as normal.
            return s_max + 1e-6

        thresholds = torch.linspace(s_min, s_max, num_bins + 2,
                                    device=scores.device)[1:-1]  # exclude endpoints

        # Vectorised Otsu: for each candidate threshold compute inter-class variance.
        # scores is 1-D, thresholds is 1-D; compare via broadcasting.
        mask = sorted_scores.unsqueeze(0) < thresholds.unsqueeze(1)  # [T, N]

        w0 = mask.sum(dim=1).float()          # count of "normal" class
        w1 = n - w0                            # count of "anomaly" class

        # Avoid division-by-zero for degenerate splits
        valid = (w0 > 0) & (w1 > 0)
        if not valid.any():
            return (s_min + s_max) / 2.0

        sum_all = sorted_scores.sum()
        # cumulative sum trick: sum of scores < threshold
        cum_sum = (sorted_scores.unsqueeze(0) * mask.float()).sum(dim=1)

        mean0 = cum_sum / w0.clamp(min=1)
        mean1 = (sum_all - cum_sum) / w1.clamp(min=1)

        sigma_b = w0 * w1 * (mean0 - mean1) ** 2  # inter-class variance

        sigma_b = sigma_b.masked_fill(~valid, -1.0)
        best_idx = sigma_b.argmax()
        return thresholds[best_idx].item()

    def gather_loss(self, query, keys, reliability=None):
        distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance, _ = torch.min(distribution, dim=2)
        if reliability is None:
            gather_loss = distance.mean()
        else:
            reliability = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
            gather_loss = (distance * reliability).sum(dim=1).mean()
        return gather_loss

    def _select_normal_frames_fixed(self, anomaly_scores, visual_features, lengths, B, N, D):
        """Original fixed-ratio normal frame selection (fallback)."""
        if lengths is not None:
            lengths_clamped = lengths.to(device=visual_features.device, dtype=torch.long).clamp(min=1, max=N)
            valid_mask = torch.arange(N, device=visual_features.device)[None, :] < lengths_clamped[:, None]
            scores = anomaly_scores.masked_fill(~valid_mask, float('inf'))
            min_valid_length = int(lengths_clamped.min().item())
        else:
            scores = anomaly_scores
            min_valid_length = N

        num_normal_frames = max(1, min(int(N * self.normal_selection_ratio), min_valid_length))
        _, indices = torch.topk(scores, k=num_normal_frames, largest=False, dim=1)
        normal_indices = indices[:, :num_normal_frames]

        selected_normal_features = torch.gather(
            visual_features, 1,
            normal_indices.unsqueeze(-1).expand(-1, -1, D)
        )
        selected_scores = torch.gather(scores, 1, normal_indices)
        reliability = (1.0 - selected_scores).clamp(min=1e-4, max=1.0)
        return selected_normal_features, reliability

    def _select_normal_frames_adaptive(self, anomaly_scores, visual_features, lengths, B, N, D):
        """Adaptive normal frame selection using Otsu's thresholding per video."""
        device = visual_features.device

        if lengths is not None:
            lengths_clamped = lengths.to(device=device, dtype=torch.long).clamp(min=1, max=N)
            valid_mask = torch.arange(N, device=device)[None, :] < lengths_clamped[:, None]
        else:
            lengths_clamped = torch.full((B,), N, device=device, dtype=torch.long)
            valid_mask = torch.ones(B, N, device=device, dtype=torch.bool)

        # --- Per-video Otsu thresholding ---
        per_video_counts = []
        normal_mask = torch.zeros(B, N, device=device, dtype=torch.bool)

        for i in range(B):
            L_i = int(lengths_clamped[i].item())
            valid_scores_i = anomaly_scores[i, :L_i]

            threshold_i = self.otsu_threshold_1d(valid_scores_i)

            # Mark frames below threshold as normal
            frame_normal = anomaly_scores[i, :L_i] < threshold_i

            count_i = int(frame_normal.sum().item())

            # Apply safety bounds
            min_k = min(self.min_normal_frames, L_i)
            max_k = max(min_k, int(self.max_normal_ratio * L_i))
            count_i = max(min_k, min(count_i, max_k))

            # If Otsu selected too few or too many, fall back to topk
            if int(frame_normal.sum().item()) < min_k or int(frame_normal.sum().item()) > max_k:
                _, topk_idx = torch.topk(anomaly_scores[i, :L_i], k=count_i, largest=False)
                normal_mask[i, :L_i] = False
                normal_mask[i, topk_idx] = True
            else:
                # Take exactly count_i frames with lowest scores among normal-flagged ones
                scores_masked = anomaly_scores[i].clone()
                scores_masked[~frame_normal.new_zeros(N, dtype=torch.bool).scatter_(0, torch.arange(L_i, device=device)[frame_normal], True)] = float('inf')
                _, topk_idx = torch.topk(scores_masked, k=count_i, largest=False)
                normal_mask[i] = False
                normal_mask[i, topk_idx] = True

            per_video_counts.append(count_i)

        # --- Pad selected features into batched tensor ---
        max_count = max(per_video_counts)
        selected_normal_features = torch.zeros(B, max_count, D, device=device, dtype=visual_features.dtype)
        selected_scores = torch.ones(B, max_count, device=device)  # default 1.0 → reliability ~0
        selection_mask = torch.zeros(B, max_count, device=device, dtype=torch.bool)

        for i in range(B):
            idx_i = normal_mask[i].nonzero(as_tuple=False).squeeze(-1)
            k_i = per_video_counts[i]
            selected_normal_features[i, :k_i] = visual_features[i, idx_i[:k_i]]
            selected_scores[i, :k_i] = anomaly_scores[i, idx_i[:k_i]]
            selection_mask[i, :k_i] = True

        reliability = (1.0 - selected_scores).clamp(min=1e-4, max=1.0)
        # Zero-out reliability for padded positions
        reliability = reliability * selection_mask.float()

        return selected_normal_features, reliability

    def forward(self, visual_features, logits1, lengths=None, normal_selection_ratio=0.125):
        B, N, D = visual_features.shape

        with torch.no_grad():
            anomaly_scores = torch.sigmoid(logits1.squeeze(-1))

            if self.adaptive_normal_selection:
                selected_normal_features, reliability = self._select_normal_frames_adaptive(
                    anomaly_scores, visual_features, lengths, B, N, D
                )
            else:
                selected_normal_features, reliability = self._select_normal_frames_fixed(
                    anomaly_scores, visual_features, lengths, B, N, D
                )

        # Reliability only modulates the normal evidence. The prototypes remain
        # video-specific because every global seed is conditioned on this video's
        # selected normal snippets by the aggregation blocks below.
        weighted_normal_features = selected_normal_features * (0.5 + 0.5 * reliability.unsqueeze(-1))

        agg_prototype = self.video_prototypes.unsqueeze(0).expand(B, -1, -1)

        for blk in self.dnp_extractor:
            agg_prototype = blk(agg_prototype, weighted_normal_features)

        dynamic_normal_patterns = agg_prototype

        g_loss = self.gather_loss(selected_normal_features, dynamic_normal_patterns, reliability)

        bottleneck_features = visual_features
        for blk in self.bottleneck:
            bottleneck_features = blk(bottleneck_features)
        reconstructed_features = bottleneck_features
        for blk in self.decoder:
            reconstructed_features = blk(reconstructed_features, dynamic_normal_patterns)

        return reconstructed_features, dynamic_normal_patterns, g_loss

class DSANet(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 args,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.rupa_use = getattr(args, 'rupa_use', True)
        self.routing_det_weight = getattr(args, 'routing_det_weight', 0.5)
        self.routing_rec_weight = getattr(args, 'routing_rec_weight', 0.3)
        self.routing_sem_weight = getattr(args, 'routing_sem_weight', 0.2)
        routing_weight_sum = (
            self.routing_det_weight + self.routing_rec_weight + self.routing_sem_weight
        )
        if routing_weight_sum <= 0:
            raise ValueError('At least one RUPA routing weight must be positive.')
        self.routing_det_weight /= routing_weight_sum
        self.routing_rec_weight /= routing_weight_sum

        self.routing_sem_weight /= routing_weight_sum
        
        self.routing_mode = getattr(args, 'routing_mode', 'safe_gate')
        if self.routing_mode not in {'fixed', 'safe_gate'}:
            raise ValueError("routing_mode must be 'fixed' or 'safe_gate'.")
        self.routing_gate_init = float(getattr(args, 'routing_gate_init', 0.01))
        self.routing_gate_max = float(getattr(args, 'routing_gate_max', 1.0))
        routing_gate_hidden = int(getattr(args, 'routing_gate_hidden', 8))


        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        mlp_dropout = getattr(args, 'mlp_dropout', 0.1)
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(mlp_dropout)),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(mlp_dropout)),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False


        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)

        self.clip_adapter = CLIP_Adapter(self.clipmodel, self.device, args.text_adapt_until, args.t_w)

        
        self.video_anomaly_refiner = SGNM(
            feature_dim=visual_width,
            num_prototypes=args.num_prototypes,
            num_heads=8,
            normal_selection_ratio=args.normal_selection_ratio,
            adaptive_normal_selection=getattr(args, 'adaptive_normal_selection', True),
            min_normal_frames=getattr(args, 'min_normal_frames', 4),
            max_normal_ratio=getattr(args, 'max_normal_ratio', 0.8),
        )

        self.rupa_gate = nn.Sequential(
            nn.Linear(5, routing_gate_hidden),
            QuickGELU(),
            nn.Linear(routing_gate_hidden, 1),
        )

        self._text_features_cache = None

        
        self.initialize_parameters()
        nn.init.normal_(self.rupa_gate[0].weight, std=0.01)
        nn.init.constant_(self.rupa_gate[0].bias, 0)
        nn.init.constant_(self.rupa_gate[-1].weight, 0)
        gate_logit = math.log(self.routing_gate_init / (1.0 - self.routing_gate_init))
        nn.init.constant_(self.rupa_gate[-1].bias, gate_logit)


    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

        trainable_modules = nn.ModuleList([self.video_anomaly_refiner])
        for m in trainable_modules.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def build_attention_mask(self, attn_window):
        mask = torch.ones(self.visual_length, self.visual_length, dtype=torch.bool)
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = False
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = False

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        if padding_mask is None and lengths is not None:
            padding_mask = (
                torch.arange(self.visual_length, device=images.device)[None, :]
                >= lengths.to(images.device)[:, None]
            )
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, padding_mask))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def get_text_features(self, text):
        if not self.training and self._text_features_cache is not None:
            return self._text_features_cache

        category_features = []
        if len(text) == 14:
            DESCRIPTIONS = DESCRIPTIONS_ORI
        else:
            DESCRIPTIONS = DESCRIPTIONS_ORI_XD
        for class_name, descriptions in DESCRIPTIONS.items():
            tokens = clip.tokenize(descriptions).to(self.device)

            text_features = self.clip_adapter.encode_text(tokens)
            mean_feature = text_features.mean(dim=0)
            mean_feature = mean_feature / mean_feature.norm()
            category_features.append(mean_feature)
        text_features_ori = torch.stack(category_features, dim=0)

        if not self.training:
            self._text_features_cache = text_features_ori

        return text_features_ori
    #thứ tự đầu vào của model
    def forward(self, visual, padding_mask, text, lengths, DNP_use, scale = 10):
        #NHÁNH 2: phát hiện bắt thuơgnf

        #visual:gồm toàn bộ feature của video sau padding VD:shape visual1[32,256,512]
        #padding_mask: báo cho model biết phần nào tensor là data thật, data giả (chèn vector 0 vào cho đủ 256 segment)
            #vì model nhìn từ segment này sang segment khác nên nếu không báo thì dẫn đến attention bị nhiễu -> feature/anomaly score sai
        #lenghts: độ dài segment
        visual_features = self.encode_video(visual, padding_mask, lengths)#đã học được ngữ cảnh của thời gian(segment này học sang segment khác)

        #mlp2: mạng Multi Layer Perceptron: chuẩn bị feature cho anomaly detection(visual_features:có người có chuyển động)
        #output = visual_features + MLP(visual_features) -> giữ info cũ học thêm info mới (chú ý vào chuyển động mạnh, tách background)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features)) # cho biết segment nào đáng chú ý
        #visual_features[32,256,512] -> logits1[32,256,1]
        #1: một điểm anomaly cho mỗi segment

        #đưa các text_prompt qua CLIP encoder [14, 512] : 512 chiều của 13 anomaly & 1 normal
        text_features_ori = self.get_text_features(text)
        text_features = text_features_ori

        #đổi chiều logits1 để nhân ma trận [32, 512, 1] -> [32, 1, 512]
        logits_attn = logits1.permute(0, 2, 1) #gom các segment đáng chú ý thành một vector video

        visual_attn = logits_attn @ visual_features #dùng vector đó để điều chỉnh text feature
        #[32, 256, 512] * [32, 1, 256] = [32, 1, 256]
        #visual_attn = [32, 1, 256]: 1 video chauws 1 segment có 256 điểm bất thường [x1,x2,x3,x4,...,x256] (đã one hot 0-1)
        #giar sử video có: segment 1: đi bộ → score thấp, segment 4: đánh nhau → score cao
        #nhân với segment: feature đi bộ × trọng số nhỏ, feature đánh nhau × trọng số lớn -> kết quả nghiêng về anomaly figting


        #Chuẩn hóa vector video(giảm khoảng cách của vector visual_attn = 1) để dùng ngữ nghĩa của vector này mà không bị ảnh hưởng bởi độ dài -> tránh lấn át text_feature
        #chia từng số trong vector visual_attn cho visual_attn.norm(kc euclid)
        visual_attn = F.normalize(visual_attn, dim=-1)

        #sao chép vector video cho 14 lớp
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        #visual_attn [32,1,512] -> [32,14,512]


        #thêm chiều batchsize cho text_features để cùng dạng vector sau đó + visual_attn
        text_features = text_features_ori.unsqueeze(0)

        #expand mở rộng chiều batch bằng với visual_attn
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])

        #dùng phần chú ý(visual_attn) để thay đổi text embedding -> model so với từng segment bị chú ý
        #text "normal"   + vector video 1
        #text "abuse"    + vector video 1...
        text_features = text_features + visual_attn

        #cho mlp1 xử lý biến đổi từng vector 512 chiều
        text_features = text_features + self.mlp1(text_features)


        #chuẩn hóa timestep của video về = 1 như trên
        visual_features_norm = F.normalize(visual_features, dim=-1)
        #chuẩn hóa text_feature
        text_features_norm = F.normalize(text_features, dim=-1)
        text_features_norm = text_features_norm.permute(0, 2, 1)


        #/0.07 để logits2 lớn hơn -> hàm loss phân loớp rõ ràng hơn
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        # RUPA-DSA: reconstruct the counterfactual normal component first, then
        # interpret the reconstruction residual with fine-grained anomaly text.
        DNP = None
        routing_prob = torch.sigmoid(logits1)
        event_features = visual_features
        background_features = visual_features
        if DNP_use:
            reconstructed_features, dynamic_normal_patterns, g_loss = self.video_anomaly_refiner(
                visual_features, logits1, lengths
            )
            residual_features = visual_features - reconstructed_features

            DNP = {
                'original_features': visual_features,
                'reconstructed_features': reconstructed_features,
                'residual_features': residual_features,
                'dynamic_normal_patterns': dynamic_normal_patterns,
                'g_loss': g_loss,
            }

            if self.rupa_use:
                category_text_norm = F.normalize(text_features_ori, dim=-1)
                category_text_matrix = category_text_norm.t().to(visual_features.dtype)
                reconstructed_norm = F.normalize(reconstructed_features, dim=-1)
                residual_norm = F.normalize(residual_features, dim=-1)

                reconstructed_semantic_logits = reconstructed_norm @ category_text_matrix / 0.07
                residual_semantic_logits = residual_norm @ category_text_matrix / 0.07

                # Normal is represented by F_rec; anomaly categories are represented
                # by R = F_video - F_rec. This is the per-snippet semantic view used
                # both for training and fine-grained inference.
                semantic_logits = torch.cat(
                    [reconstructed_semantic_logits[:, :, :1], residual_semantic_logits[:, :, 1:]],
                    dim=-1,
                )
                semantic_score = 1.0 - F.softmax(semantic_logits, dim=-1)[:, :, :1]
                reconstruction_score = (
                    1.0 - F.cosine_similarity(
                        visual_features, reconstructed_features, dim=-1
                    )
                ).unsqueeze(-1) / 2.0
                reconstruction_score = reconstruction_score.clamp(0.0, 1.0)
                detector_score = torch.sigmoid(logits1)
                fixed_routing_prob = (
                    self.routing_det_weight * detector_score
                    + self.routing_rec_weight * reconstruction_score
                    + self.routing_sem_weight * semantic_score
                ).clamp(1e-5, 1.0 - 1e-5)
                
                if self.routing_mode == 'safe_gate':
                    gate_inputs = torch.cat([
                        detector_score, reconstruction_score, semantic_score,
                        (reconstruction_score - detector_score).abs(),
                        (semantic_score - detector_score).abs()
                    ], dim=-1)
                    routing_gate = torch.sigmoid(self.rupa_gate(gate_inputs)) * self.routing_gate_max
                    routing_prob = detector_score + routing_gate * (fixed_routing_prob - detector_score)
                    routing_prob = routing_prob.clamp(1e-5, 1.0 - 1e-5)
                else:
                    routing_prob = fixed_routing_prob

                normal_text = category_text_norm[0].to(
                    dynamic_normal_patterns.dtype
                ).view(1, 1, -1)
                dnp_normal_similarity = F.cosine_similarity(
                    dynamic_normal_patterns, normal_text, dim=-1
                )

                DNP.update({
                    'semantic_logits': semantic_logits,
                    'residual_semantic_logits': residual_semantic_logits,
                    'reconstructed_semantic_logits': reconstructed_semantic_logits,
                    'semantic_scores': semantic_score,
                    'reconstruction_scores': reconstruction_score,
                    'routing_scores': routing_prob,
                    'routing_logits': torch.logit(routing_prob),
                    'dnp_normal_loss': (1.0 - dnp_normal_similarity).mean(),
                })
                event_features = residual_features
                background_features = reconstructed_features

        #chuyển logits1 sang xác suất xảy ra bất thường
        #σ(x)=1/(1+e^-x​)
        logits = routing_prob

        valid_mask = (
            torch.arange(visual_features.shape[1], device=visual_features.device)[None, :]
            < lengths.to(visual_features.device)[:, None]
        ).unsqueeze(-1).to(visual_features.dtype)

        #tạo trọng số cho hành động anomaly
        #VD:scale = 10 có xác suất p1, p2 = 0.1, 0.9 | p1: exp(10 × 0.1) - 1 = 1.718| p2: exp(10 × 0.9) - 1 = 8102
        #Segment có anomaly trọng số cao được nhấn mạnh -> khuyeechs đại hành động hơi bất thường thành cực kì bất thường
        abn_logits = ((scale * logits).exp() - 1) * valid_mask
        #Chuẩn hóa trọng số bất thường
        abn_logits = F.normalize(abn_logits, p=1, dim=1) #segment đại diện cho anomaly

        #tương tự tạo trọng số cho phần bình thường
        nor_logits = ((scale * (1. - logits)).exp() - 1) * valid_mask #1 - anomaly = normal
        nor_logits = F.normalize(nor_logits, p=1, dim=1) #segment đại diện cho normal
        #logits1:
        #đoạn nào bất thường?
        #visual_attn:
        #tóm tắt các nội dung được logits1 chú ý
        #text_features:
        #điều chỉnh ý nghĩa các lớp theo video hiện tại
        #logits2:
        #mỗi đoạn giống lớp nào nhất?

        #Tổng hợp feature bất thường
        abn_feat = torch.matmul(abn_logits.permute(0, 2, 1), event_features)
        #lưu feature bâất thường gốc
        abn_feat_ori = abn_feat #abn_feat: đại diện cho phần bất thường của video

        #Tổng hợp feature bất thường
        nor_feat = torch.matmul(nor_logits.permute(0, 2, 1), background_features)
        #lưu feature bth gốc
        nor_feat_ori = nor_feat #nor_feat: đại diện phần nền hoặc phần bình thường của video


        #khởi tạo text_features cho từng video
        nor_text_features = text_features_ori.unsqueeze(0) #thêm chiều batch
        nor_text_features = nor_text_features.expand(abn_feat.shape[0], nor_text_features.shape[1], nor_text_features.shape[2])

        #chuẩn hóa về độ dài 1
        nor_text_features_norm = F.normalize(nor_text_features, dim=-1)
        nor_text_features_norm = nor_text_features_norm.permute(0, 2, 1)

        #Chuẩn hóa feature bình thường và bất thường
        nor_visual_features_norm = F.normalize(nor_feat_ori, dim=-1)
        abn_visual_features_norm = F.normalize(abn_feat_ori, dim=-1)

        #Tính logits3(feature bất thường của toàn video giống từng lớp text đến mức nào) VD:Normal: thấp|Abuse: trung bình|Fighting: cao
        #trong hàm loss model cố làm cho abn_feat của video figting gần textfeature fighting
        logits3 = abn_visual_features_norm @ nor_text_features_norm.type(abn_visual_features_norm.dtype) / 0.07
        #tính logits4 (feature bình thường của video giống từng lớp text đến mức nào) tương tự ngược lại với bất thường
        logits4 = nor_visual_features_norm @ nor_text_features_norm.type(nor_visual_features_norm.dtype) / 0.07

        if DNP_use:
            return text_features_ori, logits1, logits2, logits3, logits4, DNP #DNP: phần nào không thể được normal prototypes tái tạo tốt?
        else:
            return text_features_ori, logits1, logits2, logits3, logits4
