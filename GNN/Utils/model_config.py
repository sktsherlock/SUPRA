import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GNN.fusions.MultiBench import *

import argparse

def str2bool(v):
    # 兼容 wandb / 命令行传参（通常是字符串）
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "t", "on"):
        return True
    if v in ("false", "0", "no", "n", "f", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def _parse_metric(metric_str):
    """Parse metric string, supporting combined format like 'f1_macro'.

    Returns (metric, average) tuple.
    """
    if metric_str is None:
        return 'accuracy', 'macro'
    metric_str = str(metric_str).strip().lower()
    _compound_metrics = {'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro',
                         'precision_micro', 'precision_weighted', 'recall_macro',
                         'recall_micro', 'recall_weighted'}
    if metric_str in _compound_metrics:
        parts = metric_str.rsplit('_', 1)
        return parts[0], parts[1]
    return metric_str, None


class _MetricAction(argparse.Action):
    """Custom action for --metric: handles 'f1_macro' style as compound metric."""
    def __call__(self, parser, namespace, values, option_string=None):
        metric_val, avg_val = _parse_metric(values)
        setattr(namespace, self.dest, metric_val)
        # If metric was compound (e.g., f1_macro), also set average
        if avg_val is not None:
            setattr(namespace, 'average', avg_val)

def add_common_args(argparser):
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument(
        "--n-runs", type=int, default=3, help="running times"
    )
    argparser.add_argument(
        "--n-epochs", type=int, default=1000, help="number of epochs"
    )
    argparser.add_argument(
        "--lr", type=float, default=0.005, help="learning rate"
    )
    argparser.add_argument(
        "--n-layers", type=int, default=3, help="number of layers"
    )
    argparser.add_argument(
        "--n-hidden", type=int, default=256, help="number of hidden units"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate"
    )
    argparser.add_argument(
        "--min-lr", type=float, default=0.0001, help="the min learning rate"
    )
    argparser.add_argument(
        "--label-smoothing", type=float, default=0.1, help="the smoothing factor"
    )
    argparser.add_argument(
        "--kd_weight", type=float, default=0.0, help="Weight for logit-level distillation loss"
    )
    argparser.add_argument(
        "--kd_temperature", type=float, default=1.0, help="Temperature for distillation"
    )
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    # ! default
    argparser.add_argument(
        "--log-every", type=int, default=20, help="log every LOG_EVERY epochs"
    )
    argparser.add_argument(
        "--eval_steps", type=int, default=1, help="eval in every epochs"
    )
    argparser.add_argument(
        "--early_stop_patience", type=int, default=None,
        help="when to stop the  training loop to be aviod of the overfiting"
    )
    argparser.add_argument(
        "--warmup_epochs", type=int, default=None, help="The warmup epochs"
    )
    argparser.add_argument(
        "--seed", type=int, default=42, help="The seed for the teacher models"
    )
    # ! Model related
    argparser.add_argument(
        "--model_name", type=str, default='GCN', help="The model name.",
    )
    argparser.add_argument(
        "--fusion_name", type=str, default='LRTF', help="The fusion model name.",
    )
    # ! Data related
    argparser.add_argument(
        "--data_name", type=str, default=None, help="The dataset name.",
    )
    argparser.add_argument(
        "--exp_path", type=str, default='Exp/Transductive/', help="The dataset name.",
    )
    argparser.add_argument(
        "--feature", type=str, default=None, help="Use Unimodal feature embedding as feature",
    )
    argparser.add_argument(
        "--text_feature", type=str, default=None, help="Use text feature embedding as feature",
    )
    argparser.add_argument(
        "--visual_feature", type=str, default=None, help="Use visual feature embedding as feature",
    )
    argparser.add_argument(
        "--graph_path", type=str, default=None, help="The datasets to be implemented."
    )
    argparser.add_argument(
        "--undirected", type=str2bool, default=True, help="Whether to undirect the graph."
    )
    argparser.add_argument(
        "--selfloop", type=str2bool, default=True, help="Whether to add self loop in the graph."
    )
    argparser.add_argument(
        "--metric", type=str, default='accuracy', action=_MetricAction,
        help="Metric to use (e.g., accuracy, f1, or compound like f1_macro)."
    )
    argparser.add_argument(
        "--average", type=str, default='macro', choices=['weighted', 'micro', 'macro', None]
    )
    argparser.add_argument(
        "--result_csv",
        type=str,
        default=None,
        help="Optional CSV path for auto-updating best results (by full metric).",
    )
    argparser.add_argument(
        "--result_csv_all",
        type=str,
        default=None,
        help="Optional CSV path for appending all results (per config).",
    )
    argparser.add_argument(
        "--result_csv_overall",
        type=str,
        default=None,
        help="Optional CSV path for auto-updating best results by overall harmonic mean.",
    )
    argparser.add_argument(
        "--result_tag",
        type=str,
        default=None,
        help="Optional method label used in result CSV (e.g., GNN, OGMGNN).",
    )
    argparser.add_argument(
        "--best_ignore_layers",
        type=str2bool,
        default=True,
        help=(
            "When updating best-result CSVs, ignore n_layers in the key so the best config "
            "is selected automatically across different layer counts."
        ),
    )
    argparser.add_argument(
        "--report_acc_f1", type=str2bool, default=True,
        help="Report Acc and Macro-F1 at the selected checkpoint."
    )
    argparser.add_argument(
        "--report_drop_modality",
        type=str2bool,
        default=False,
        help="Report metrics when dropping text or visual modality at eval time.",
    )
    argparser.add_argument(
        "--report_drop_mode",
        type=str,
        default="best",
        choices=["best", "always"],
        help="When to compute drop-modality metrics: only on new best-val or at every eval.",
    )
    argparser.add_argument(
        "--degrade_target",
        type=str,
        default="both",
        choices=["text", "visual", "both", "none"],
        help="Which modality to corrupt when reporting degrade metrics.",
    )
    argparser.add_argument(
        "--degrade_alphas",
        type=str,
        default="",
        help="Comma/space-separated noise levels for degrade metrics (e.g., '0.2,0.4,0.6,0.8,1.0').",
    )
    # NOTE: report_acc_f1 / drop-modality metrics are handled in training scripts.
    # ! Split datasets
    argparser.add_argument(
        "--inductive", type=str2bool, default=False, help="Whether to do inductive learning experiments."
    )
    argparser.add_argument(
        "--train_ratio", type=float, default=0.6, help="training ratio"
    )
    argparser.add_argument(
        "--val_ratio", type=float, default=0.2, help="val ratio"
    )
    argparser.add_argument(
        "--test_ratio", type=float, default=0.08, help="test ratio for link prediction"
    )
    argparser.add_argument(
        "--fewshots", type=int, default=None, help="fewshots values"
    )


def add_fusion_args(argparser):
    argparser.add_argument("--input_dims", nargs="+", type=int, help="Input dimensions of the modalities")
    argparser.add_argument(
        "--output_dim", type=int, default=768, help="number of output_dim"
    )
    argparser.add_argument(
        "--rank", type=int, default=10, help="a hyperparameter of LRTF. See link above for details"
    )


def add_gat_args(argparser):
    argparser.add_argument(
        "--no-attn-dst", type=str2bool, default=True, help="Don't use attn_dst."
    )
    argparser.add_argument(
        "--n-heads", type=int, default=3, help="number of heads"
    )
    argparser.add_argument(
        "--attn-drop", type=float, default=0.0, help="attention drop rate"
    )
    argparser.add_argument(
        "--edge-drop", type=float, default=0.0, help="edge drop rate"
    )


def add_sage_args(argparser):
    argparser.add_argument(
        "--aggregator", type=str, default="mean", choices=["mean", "gcn", "pool", "lstm"],
        help="Specify the aggregator option"
    )


def add_sgc_args(argparser):
    argparser.add_argument(
        "--bias", type=str2bool, default=True, help="control the SGC bias."
    )
    argparser.add_argument(
        "--k", type=int, default=2, help="number of k"
    )


def add_revgat_args(argparser):
    argparser.add_argument(
        "--use_symmetric_norm", type=str2bool, default=True, help="if False, no W."
    )


def add_appnp_args(argparser):
    argparser.add_argument(
        "--edge_drop", type=float, default=0.5, help="edge propagation dropout"
    )
    argparser.add_argument(
        "--k_ps", type=int, default=10, help="Number of propagation steps"
    )
    argparser.add_argument(
        "--alpha", type=float, default=0.1, help="Teleport Probability"
    )
    argparser.add_argument(
        "--input_dropout", type=float, default=0.5, help="dropout rate"
    )


def gen_model(args, device, n_classes, t_dim, v_dim):
    if args.model_name == 'GCN':
        from ..Library.GCN import GCN
        if args.fusion_name == 'Concat':
            model = GCN(t_dim + v_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout).to(device)
        else:
            model = GCN(args.output_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout).to(device)
    elif args.model_name == 'SAGE':
        from ..Library.GraphSAGE import GraphSAGE
        if args.fusion_name == 'Concat':
            model = GraphSAGE(t_dim + v_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout,
                              aggregator_type=args.aggregator).to(device)
        else:
            model = GraphSAGE(args.output_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout,
                              aggregator_type=args.aggregator).to(device)
    elif args.model_name == 'SGC':
        from dgl.nn.pytorch.conv import SGConv
        if args.fusion_name == 'Concat':
            model = SGConv(t_dim + v_dim, n_classes, args.k, cached=True, bias=args.bias).to(device)
        else:
            model = SGConv(args.output_dim, n_classes, args.k, cached=True, bias=args.bias).to(device)
    elif args.model_name == 'RevGAT':
        from ..Library.RevGAT.model import RevGAT
        if args.fusion_name == 'Concat':
            model = RevGAT(t_dim + v_dim, n_classes, args.n_hidden, args.n_layers, args.n_heads, F.relu,
                           dropout=args.dropout,
                           attn_drop=args.attn_drop, edge_drop=args.edge_drop, use_attn_dst=False,
                           use_symmetric_norm=args.use_symmetric_norm).to(device)
        else:
            model = RevGAT(args.output_dim, n_classes, args.n_hidden, args.n_layers, args.n_heads, F.relu,
                           dropout=args.dropout,
                           attn_drop=args.attn_drop, edge_drop=args.edge_drop, use_attn_dst=False,
                           use_symmetric_norm=args.use_symmetric_norm).to(device)
    elif args.model_name == 'APPNP':
        from ..Library.APPNP import APPNP
        if args.fusion_name == 'Concat':
            model = APPNP(t_dim + v_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.input_dropout,
                          args.edge_drop, args.alpha, args.k_ps).to(device)
        else:
            model = APPNP(args.output_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.input_dropout,
                          args.edge_drop, args.alpha, args.k_ps).to(device)
    elif args.model_name == 'GAT':
        from ..Library.GAT import GAT
        if args.fusion_name == 'Concat':
            model = GAT(t_dim + v_dim, n_classes, args.n_hidden, args.n_layers, args.n_heads, F.relu, args.dropout,
                        args.attn_drop, args.edge_drop, not args.no_attn_dst).to(device)
        else:
            model = GAT(args.output_dim, n_classes, args.n_hidden, args.n_layers, args.n_heads, F.relu, args.dropout,
                        args.attn_drop, args.edge_drop, not args.no_attn_dst).to(device)
    else:
        raise ValueError('GNN must be in the implementrary.')
    return model


def gen_fusion(args, device, t_dim, v_dim):
    if args.fusion_name == 'TF':
        fusion = TensorFusion()
    elif args.fusion_name == 'LRTF':
        fusion = LowRankTensorFusion(args.input_dims, args.output_dim, args.rank).to(device)
    elif args.fusion_name == 'ConcatWithLinear':
        fusion = ConcatWithLinear(t_dim + v_dim, args.output_dim).to(device)
    elif args.fusion_name == 'Concat':
        fusion = Concat()
    else:
        raise ValueError('Fusion must be in the implementrary.')
    return fusion
