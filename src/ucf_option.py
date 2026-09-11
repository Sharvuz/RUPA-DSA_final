import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


parser = argparse.ArgumentParser(description='DSANet')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int) #
parser.add_argument('--visual-length', default=256, type=int) #moi video tach thanh 256 doan
parser.add_argument('--visual-width', default=512, type=int) #moi video dc bieu dien bang 1 vecto CLIP 512 chieu
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int) #13-bat thuong, 1 normal
#tensor video khi -> model [Batchsize, 256, 512]


parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=str2bool)
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')

parser.add_argument('--lr', type=float, default=7e-5)

#DNP
parser.add_argument('--decoder_depth', type=int, default=8)
parser.add_argument('--normal_selection_ratio', type=float, default=0.8)
parser.add_argument('--num_prototypes', type=int, default=16)
parser.add_argument('--DNP_use', default=True, type=str2bool)

# Adaptive Normal Selection (Improvement 2)
parser.add_argument('--adaptive_normal_selection', default=True, type=str2bool,
                    help='Use Otsu thresholding for dynamic normal frame selection')
parser.add_argument('--min_normal_frames', type=int, default=4,
                    help='Minimum number of normal frames to select per video')
parser.add_argument('--max_normal_ratio', type=float, default=0.8,
                    help='Maximum ratio of frames that can be selected as normal')

#Adapter
parser.add_argument('--text_adapt_until', default=3, type=int)
parser.add_argument('--t_w', default=0.1, type=float)

parser.add_argument('--temp', default=5.0, type=float)

parser.add_argument('--loss2_weight', type=float, default=1.1)

# RUPA-DSA: counterfactual reconstruction, residual semantics and closed-loop routing.
parser.add_argument('--rupa-use', default=True, type=str2bool)
parser.add_argument('--routing-det-weight', default=0.5, type=float)
parser.add_argument('--routing-rec-weight', default=0.3, type=float)
parser.add_argument('--routing-sem-weight', default=0.2, type=float)
parser.add_argument('--loss-residual-weight', default=1.0, type=float)
parser.add_argument('--loss-reconstructed-normal-weight', default=1.0, type=float)
parser.add_argument('--loss-dnp-normal-weight', default=0.1, type=float)
parser.add_argument('--loss-consistency-weight', default=1.0, type=float)

parser.add_argument('--loss-gather-weight', default=1.0, type=float)

# Safe Gate and Warm Start
parser.add_argument('--init-model-path', default=None, type=str)
parser.add_argument('--routing-mode', choices=['fixed', 'safe_gate'], default='safe_gate', type=str)
parser.add_argument('--routing-gate-init', default=0.01, type=float)
parser.add_argument('--routing-gate-max', default=1.0, type=float)
parser.add_argument('--routing-gate-hidden', default=8, type=int)
parser.add_argument('--main-lr', default=0.0, type=float)
parser.add_argument('--refiner-lr', default=1e-5, type=float)
parser.add_argument('--loss-routing-weight', default=1.0, type=float)
parser.add_argument('--loss-routing-anchor-weight', default=0.1, type=float)

parser.add_argument('--num-workers', default=2, type=int)

# === Improvement hyperparameters (v2.1) ===
# LR scaling: scale LR by sqrt(batch_size / lr_scale_ref_bs)
parser.add_argument('--lr-scale-ref-bs', default=0, type=int,
                    help='Reference batch size for sqrt LR scaling. 0 = disabled.')
# Gradient clipping
parser.add_argument('--grad-clip', default=0.0, type=float,
                    help='Max gradient norm for clipping. 0 = disabled.')
# Early stopping
parser.add_argument('--patience', default=0, type=int,
                    help='Early stopping patience (epochs). 0 = disabled.')
# EMA
parser.add_argument('--ema-decay', default=0.0, type=float,
                    help='EMA decay rate for model weights. 0 = disabled.')
# Feature-level augmentation
parser.add_argument('--feat-dropout', default=0.0, type=float,
                    help='Feature-level dropout rate during training. 0 = disabled.')
parser.add_argument('--feat-noise', default=0.0, type=float,
                    help='Gaussian noise std for feature augmentation. 0 = disabled.')
# Eval interval
parser.add_argument('--eval-interval', default=1280, type=int,
                    help='Steps between evaluations during training.')
# Model dropout rate (for mlp1/mlp2)
parser.add_argument('--mlp-dropout', default=0.1, type=float,
                    help='Dropout rate for MLP layers in model.')
