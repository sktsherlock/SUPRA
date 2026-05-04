import argparse
import copy
import gc
import os
import sys
import time
from typing import Optional

from GNN.Utils.model_config import str2bool

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

# Allow running as a script: `python GNN/Baselines/Early_GNN.py`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(_ROOT))

from GNN.GraphData import load_data, set_seed  # noqa: E402
from GNN.Utils.NodeClassification import (
    mag_classification,
    _compute_degrade_metrics_mag,
    _as_scalar_float,
    _parse_degrade_alphas,
    _alpha_tag,
)  # noqa: E402
from GNN.Utils.LossFunction import cross_entropy, get_metric, EarlyStopping, adjust_learning_rate  # noqa: E402
from GNN.Utils.model_config import (  # noqa: E402
    add_common_args,
    add_sage_args,
    add_gat_args,
    add_revgat_args,
    add_sgc_args,
    add_appnp_args,
)
from GNN.Utils.result_logger import build_result_row, update_best_result_csv, append_result_csv  # noqa: E402


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


def _info_nce(z1: th.Tensor, z2: th.Tensor, temperature: float = 0.1) -> th.Tensor:
    if z1.numel() == 0:
        return th.tensor(0.0, device=z1.device)
    if z1.shape != z2.shape:
        raise ValueError(f"InfoNCE requires same shapes, got {tuple(z1.shape)} vs {tuple(z2.shape)}")
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    logits = (z1 @ z2.t()) / float(temperature)
    labels = th.arange(int(z1.shape[0]), device=z1.device, dtype=th.long)
    loss12 = F.cross_entropy(logits, labels)
    loss21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss12 + loss21)


def _kd_loss(teacher_logits: th.Tensor, student_logits: th.Tensor, temperature: float = 1.0) -> th.Tensor:
    if teacher_logits.numel() == 0:
        return teacher_logits.new_zeros(())
    t = float(temperature)
    if t <= 0.0:
        t = 1.0
    teacher_prob = F.softmax(teacher_logits / t, dim=-1)
    student_log_prob = F.log_softmax(student_logits / t, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)


def _sample_pair_idx(idx: th.Tensor, max_pairs: int) -> th.Tensor:
    if max_pairs <= 0 or int(idx.numel()) <= max_pairs:
        return idx
    perm = th.randperm(int(idx.numel()), device=idx.device)
    return idx[perm[:max_pairs]]


class SimpleMAGGNN(nn.Module):
    def __init__(
        self,
        text_encoder: Optional[nn.Module],
        visual_encoder: Optional[nn.Module],
        gnn: nn.Module,
        early_fuse: str = "concat",
        single_modality: Optional[str] = None,
        use_mlp_projection: bool = False,
        use_no_encoder: bool = False,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.gnn = gnn
        self.early_fuse = str(early_fuse).lower().strip() if early_fuse is not None else "concat"
        self.single_modality = single_modality
        self.use_mlp_projection = use_mlp_projection
        self.use_no_encoder = use_no_encoder
        if use_mlp_projection and not use_no_encoder:
            enc_out = text_encoder.proj.out_features
            self.mlp_proj = nn.Sequential(
                nn.Linear(enc_out * 2, enc_out),
                nn.ReLU(),
                nn.LayerNorm(enc_out),
                nn.Linear(enc_out, enc_out),
            )

    def reset_parameters(self):
        if self.text_encoder is not None and hasattr(self.text_encoder, "reset_parameters"):
            self.text_encoder.reset_parameters()
        if self.visual_encoder is not None and hasattr(self.visual_encoder, "reset_parameters"):
            self.visual_encoder.reset_parameters()
        if hasattr(self.gnn, "reset_parameters"):
            self.gnn.reset_parameters()

    def forward(self, graph, text_feature: th.Tensor, visual_feature: th.Tensor) -> th.Tensor:
        if self.single_modality == "text":
            feat = text_feature if self.use_no_encoder else self.text_encoder(text_feature)
        elif self.single_modality == "visual":
            feat = visual_feature if self.use_no_encoder else self.visual_encoder(visual_feature)
        elif self.use_no_encoder:
            # No encoder: raw concat → GNN (traditional GNN approach for multimodal)
            feat = th.cat([text_feature, visual_feature], dim=1)
        else:
            text_h = self.text_encoder(text_feature)
            vis_h = self.visual_encoder(visual_feature)
            if self.early_fuse == "sum":
                feat = text_h + vis_h
            else:
                feat = th.cat([text_h, vis_h], dim=1)
            if self.use_mlp_projection:
                feat = self.mlp_proj(feat)
        gnn_name = type(self.gnn).__name__
        if gnn_name == "GCNII":
            return self.gnn(feat, graph)
        return self.gnn(graph, feat)


class SimpleMAGMLP(nn.Module):
    def __init__(
        self,
        text_encoder: nn.Module,
        visual_encoder: nn.Module,
        mlp: nn.Module,
        early_fuse: str = "concat",
        single_modality: Optional[str] = None,
        use_no_encoder: bool = False,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.mlp = mlp
        self.early_fuse = str(early_fuse).lower().strip() if early_fuse is not None else "concat"
        self.single_modality = single_modality
        self.use_no_encoder = use_no_encoder

    def reset_parameters(self):
        if hasattr(self.text_encoder, "reset_parameters"):
            self.text_encoder.reset_parameters()
        if hasattr(self.visual_encoder, "reset_parameters"):
            self.visual_encoder.reset_parameters()
        if hasattr(self.mlp, "reset_parameters"):
            self.mlp.reset_parameters()

    def forward(self, graph, text_feature: th.Tensor, visual_feature: th.Tensor) -> th.Tensor:
        if self.single_modality == "text":
            feat = text_feature if self.use_no_encoder else self.text_encoder(text_feature)
        elif self.single_modality == "visual":
            feat = visual_feature if self.use_no_encoder else self.visual_encoder(visual_feature)
        elif self.use_no_encoder:
            # No encoder: raw concat → MLP (for bimodal, when use_no_encoder=True)
            feat = th.cat([text_feature, visual_feature], dim=1)
        else:
            text_h = self.text_encoder(text_feature)
            vis_h = self.visual_encoder(visual_feature)
            if self.early_fuse == "sum":
                feat = text_h + vis_h
            else:
                feat = th.cat([text_h, vis_h], dim=1)
        return self.mlp(feat)


class _SeparateHeadWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, head: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.head = head

        # For downstream analysis parity.
        self._poly_base = base_model
        self._poly_head = head

    @property
    def text_encoder(self):
        return getattr(self.base_model, "text_encoder")

    @property
    def visual_encoder(self):
        return getattr(self.base_model, "visual_encoder")

    def reset_parameters(self):
        if hasattr(self.base_model, "reset_parameters"):
            self.base_model.reset_parameters()
        if hasattr(self.head, "reset_parameters"):
            self.head.reset_parameters()

    def forward_with_drop(self, graph, text_feature: th.Tensor, visual_feature: th.Tensor, *, drop_text=False, drop_visual=False):
        if not hasattr(self.base_model, "forward_with_drop"):
            raise AttributeError("base_model must implement forward_with_drop when using separate head")
        h = self.base_model.forward_with_drop(graph, text_feature, visual_feature, drop_text=drop_text, drop_visual=drop_visual)
        return self.head(h)

    def forward(self, graph, text_feature: th.Tensor, visual_feature: th.Tensor) -> th.Tensor:
        h = self.base_model(graph, text_feature, visual_feature)
        return self.head(h)


def _make_observe_graph_inductive(graph, val_idx: th.Tensor, test_idx: th.Tensor):
    """Remove edges touching val/test nodes while keeping node IDs stable."""
    observe_graph = copy.deepcopy(graph)
    isolated = th.cat((val_idx, test_idx)).to(th.long)
    if isolated.numel() == 0:
        return observe_graph
    isolated = th.unique(isolated)

    n = observe_graph.num_nodes()
    is_isolated = th.zeros((n,), dtype=th.bool)
    is_isolated[isolated] = True

    src, dst = observe_graph.all_edges(order="eid")
    keep = ~(is_isolated[src] | is_isolated[dst])
    if bool(keep.all().item()):
        return observe_graph

    eids = th.arange(observe_graph.num_edges(), dtype=th.int64)
    remove_eids = eids[~keep]
    observe_graph = observe_graph.remove_edges(remove_eids)
    return observe_graph


def _build_gnn_backbone(args, in_dim: int, n_classes: int, device: th.device):
    name = str(getattr(args, "model_name", "GCN"))
    if name == "GCN":
        from GNN.Library.GCN import GCN

        return GCN(in_dim, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout).to(device)

    if name == "SAGE":
        from GNN.Library.GraphSAGE import GraphSAGE

        return GraphSAGE(
            in_dim,
            args.n_hidden,
            n_classes,
            args.n_layers,
            F.relu,
            args.dropout,
            aggregator_type=getattr(args, "aggregator", "mean"),
        ).to(device)

    if name == "GAT":
        from GNN.Library.GAT import GAT

        return GAT(
            in_dim,
            n_classes,
            args.n_hidden,
            args.n_layers,
            args.n_heads,
            F.relu,
            args.dropout,
            args.attn_drop,
            args.edge_drop,
            not getattr(args, "no_attn_dst", True),
        ).to(device)

    if name == "RevGAT":
        from GNN.Library.RevGAT.model import RevGAT

        return RevGAT(
            in_dim,
            n_classes,
            args.n_hidden,
            args.n_layers,
            args.n_heads,
            F.relu,
            dropout=args.dropout,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            use_attn_dst=False,
            use_symmetric_norm=getattr(args, "use_symmetric_norm", True),
        ).to(device)

    if name == "SGC":
        from dgl.nn.pytorch.conv import SGConv

        return SGConv(in_dim, n_classes, args.k, cached=True, bias=args.bias).to(device)

    if name == "APPNP":
        from GNN.Library.APPNP import APPNP

        return APPNP(
            in_dim,
            args.n_hidden,
            n_classes,
            args.n_layers,
            F.relu,
            getattr(args, "input_dropout", 0.5),
            getattr(args, "edge_drop", 0.5),
            getattr(args, "alpha", 0.1),
            getattr(args, "k_ps", 10),
        ).to(device)

    if name == "GCNII":
        from GNN.Library.GCNII import GCNII as GCNIIModel

        return GCNIIModel(
            nfeat=in_dim,
            nlayers=args.n_layers,
            nhidden=args.n_hidden,
            nclass=n_classes,
            dropout=args.dropout,
            lamda=float(getattr(args, "gcnii_lamda", 0.5)),
            alpha=float(getattr(args, "gcnii_alpha", 0.5)),
            variant=bool(getattr(args, "gcnii_variant", False)),
        ).to(device)

    if name == "JKNet":
        from GNN.Library.JKNet import JKNet as JKNetModel

        return JKNetModel(
            in_feats=in_dim,
            n_hidden=args.n_hidden,
            n_classes=n_classes,
            n_layers=args.n_layers,
            dropout=args.dropout,
            aggr=str(getattr(args, "jknet_aggr", "concat")),
        ).to(device)

    raise ValueError(f"Unsupported --model_name: {name}")


def args_init():
    parser = argparse.ArgumentParser(
        "MAG Early Fusion GNN: per-modality encoders + concat -> GNN/MLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    add_sage_args(parser)
    add_gat_args(parser)
    add_revgat_args(parser)
    add_sgc_args(parser)
    add_appnp_args(parser)

    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging for offline runs")
    parser.add_argument("--backend", type=str, default="gnn", choices=["gnn", "mlp"], help="Downstream model")
    parser.add_argument(
        "--degrade_alpha",
        type=float,
        default=1.0,
        help="Noise strength for modality degradation at eval (0=no degrade, 1=full noise).",
    )
    parser.add_argument(
        "--early_fuse",
        type=str,
        default="concat",
        choices=["concat", "sum"],
        help="Early fusion operator after per-modality encoders. 'concat' keeps 2*proj_dim, 'sum' uses proj_dim.",
    )
    parser.add_argument(
        "--early_embed_dim",
        type=int,
        default=None,
        help="Embedding dim for analysis/probing when using --separate_classifier (default: n_hidden).",
    )
    parser.add_argument(
        "--early_mlp_projection",
        type=str2bool,
        default=False,
        help="Add MLP projection (Linear→ReLU→LN→Linear) between concat and GNN.",
    )
    parser.add_argument(
        "--early_no_encoder",
        type=str2bool,
        default=True,
        help="Skip per-modality encoders and directly concat raw features to GNN (default: True).",
    )
    parser.add_argument(
        "--separate_classifier",
        action="store_true",
        help="If set, train a model that outputs an embedding (dim=early_embed_dim) and applies a separate linear head. "
             "This produces an analysis-friendly representation space (early/h).",
    )
    parser.add_argument("--mm_proj_dim", type=int, default=None, help="Per-modality encoder output dim")
    parser.add_argument(
        "--single_modality",
        type=str,
        default=None,
        choices=["text", "visual"],
        help="If set, only use the specified modality (for plain/unimodal experiments).",
    )
    parser.add_argument("--mmcl_weight", type=float, default=0.0, help="Weight for modality contrastive loss")
    parser.add_argument("--mmcl_temperature", type=float, default=0.1, help="Temperature for modality contrastive loss")
    parser.add_argument("--mmcl_max_pairs", type=int, default=2048, help="Max train nodes for contrastive loss")
    parser.add_argument(
        "--save_best_ckpt",
        type=str,
        default=None,
        help="Optional path to save best checkpoint per run (weights-only + args). If multiple runs, appends _run{k}.pt.",
    )
    parser.add_argument("--export_predictions", type=str, default=None,
                        help="Path to save test predictions as torch.Tensor (argmax, shape=[N_test])")

    # GCNII arguments
    parser.add_argument("--gcnii_lamda", type=float, default=0.5, help="GCNII damping factor (lambda)")
    parser.add_argument("--gcnii_alpha", type=float, default=0.5, help="GCNII initial residual coefficient (alpha)")
    parser.add_argument("--gcnii_variant", action="store_true", help="Use GCNII variant with concatenated initial features")

    # JKNet arguments
    parser.add_argument(
        "--jknet_aggr",
        type=str,
        default="concat",
        choices=["concat", "max", "last"],
        help="JKNet aggregation strategy: concat (default), max pooling, or last layer only",
    )

    return parser


def _build_run_ckpt_path(base_path: str, run_idx: int) -> str:
    base_path = str(base_path)
    root, ext = os.path.splitext(base_path)
    if ext.lower() != ".pt":
        ext = ".pt"
    if run_idx <= 1:
        return f"{root}{ext}"
    return f"{root}_run{run_idx}{ext}"


def _mag_classification_mmcl(
    args,
    graph,
    observe_graph,
    model,
    text_feat,
    visual_feat,
    labels,
    train_idx,
    val_idx,
    test_idx,
    n_running,
    return_extra=False,
):
    if args.early_stop_patience is not None:
        stopper = EarlyStopping(patience=args.early_stop_patience)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    try:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=100,
            verbose=True,
            min_lr=args.min_lr,
        )
    except TypeError:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=100,
            min_lr=args.min_lr,
        )

    total_time = 0
    best_val_result, final_test_result = 0.0, 0.0
    best_val_score = -1.0
    select_metric = getattr(args, "metric", "accuracy")
    select_average = getattr(args, "average", None)
    report_drop = bool(getattr(args, "report_drop_modality", False))
    report_drop_mode = str(getattr(args, "report_drop_mode", "best")).lower()
    degrade_target = str(getattr(args, "degrade_target", "both")).lower()
    degrade_alphas = _parse_degrade_alphas(args)
    best_test_degrade = None
    best_state_dict = None
    best_test_logits = None

    # Peak memory tracking — rely on PyTorch's native peak tracker.
    # NOTE: max_memory_allocated does NOT include allocator cached/reserved memory.
    peak_memory_mb = 0.0
    model_device = next(model.parameters()).device
    is_cuda_run = th.cuda.is_available() and (model_device.type == "cuda")
    if is_cuda_run:
        gc.collect()  # free Python references to prior-run tensors
        th.cuda.empty_cache()
        th.cuda.synchronize()
        th.cuda.reset_peak_memory_stats(model_device)
    epochs_needed = args.n_epochs  # will be updated if early stop triggered

    mmcl_weight = float(getattr(args, "mmcl_weight", 0.0))
    mmcl_temperature = float(getattr(args, "mmcl_temperature", 0.1))
    mmcl_max_pairs = int(getattr(args, "mmcl_max_pairs", 2048))
    kd_weight = float(getattr(args, "kd_weight", 0.0))
    kd_temperature = float(getattr(args, "kd_temperature", 1.0))

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        if args.warmup_epochs is not None:
            adjust_learning_rate(optimizer, args.lr, epoch, args.warmup_epochs)

        model.train()
        optimizer.zero_grad()
        pred = model(observe_graph, text_feat, visual_feat)
        cls_loss = cross_entropy(pred[train_idx], labels[train_idx], label_smoothing=args.label_smoothing)

        con_loss = th.tensor(0.0, device=cls_loss.device)
        if mmcl_weight > 0.0:
            text_h = model.text_encoder(text_feat)
            vis_h = model.visual_encoder(visual_feat)
            pair_idx = _sample_pair_idx(train_idx, mmcl_max_pairs)
            if int(pair_idx.numel()) > 1:
                con_loss = _info_nce(text_h[pair_idx], vis_h[pair_idx], temperature=mmcl_temperature)
        kd_loss = th.tensor(0.0, device=cls_loss.device)
        if kd_weight > 0.0:
            teacher_logits = pred.detach()
            logits_text = model.forward_with_drop(observe_graph, text_feat, visual_feat, drop_text=False, drop_visual=True)
            logits_vis = model.forward_with_drop(observe_graph, text_feat, visual_feat, drop_text=True, drop_visual=False)
            kd_loss = 0.5 * (
                _kd_loss(teacher_logits[train_idx], logits_text[train_idx], temperature=kd_temperature)
                + _kd_loss(teacher_logits[train_idx], logits_vis[train_idx], temperature=kd_temperature)
            )
        loss = cls_loss + mmcl_weight * con_loss + kd_weight * kd_loss
        loss.backward()
        optimizer.step()

        if epoch % args.eval_steps == 0:
            model.eval()
            with th.no_grad():
                pred = model(graph, text_feat, visual_feat)
            val_loss = cross_entropy(pred[val_idx], labels[val_idx], label_smoothing=args.label_smoothing)
            test_loss = cross_entropy(pred[test_idx], labels[test_idx], label_smoothing=args.label_smoothing)

            train_result = get_metric(th.argmax(pred[train_idx], dim=1), labels[train_idx], args.metric, average=args.average)
            val_result = get_metric(th.argmax(pred[val_idx], dim=1), labels[val_idx], args.metric, average=args.average)
            test_result = get_metric(th.argmax(pred[test_idx], dim=1), labels[test_idx], args.metric, average=args.average)

            if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                wandb.log(
                    {
                        "Train_loss": _as_scalar_float(loss),
                        "Val_loss": _as_scalar_float(val_loss),
                        "Test_loss": _as_scalar_float(test_loss),
                        "Train_result": _as_scalar_float(train_result),
                        "Val_result": _as_scalar_float(val_result),
                        "Test_result": _as_scalar_float(test_result),
                        "Train_contrastive_loss": _as_scalar_float(con_loss),
                        "Train_kd_loss": _as_scalar_float(kd_loss),
                    }
                )
            lr_scheduler.step(_as_scalar_float(loss))

            val_pred = th.argmax(pred[val_idx], dim=1)
            val_true = labels[val_idx]
            val_score = get_metric(val_pred, val_true, select_metric, average=select_average)

            degrade_vals = None
            if report_drop and report_drop_mode == "always":
                degrade_vals = {}
                for alpha in degrade_alphas:
                    test_degrade_text, test_degrade_vis = _compute_degrade_metrics_mag(
                        model,
                        graph,
                        text_feat,
                        visual_feat,
                        labels,
                        test_idx,
                        args.metric,
                        args.average,
                        train_idx=train_idx,
                        degrade_alpha=alpha,
                        degrade_target=degrade_target,
                    )
                    degrade_vals[alpha] = (test_degrade_text, test_degrade_vis)
                    tag = _alpha_tag(alpha)
                    if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                        log_payload = {}
                        if test_degrade_text is not None:
                            log_payload[f"Test_degrade_text_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_text)
                        if test_degrade_vis is not None:
                            log_payload[f"Test_degrade_visual_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_vis)
                        if log_payload:
                            wandb.log(log_payload)

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_result = val_result
                final_test_result = test_result
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_test_logits = pred[test_idx].detach().clone()
                if report_drop:
                    if report_drop_mode == "best":
                        best_test_degrade = {}
                        for alpha in degrade_alphas:
                            test_degrade_text, test_degrade_vis = _compute_degrade_metrics_mag(
                                model,
                                graph,
                                text_feat,
                                visual_feat,
                                labels,
                                test_idx,
                                args.metric,
                                args.average,
                                train_idx=train_idx,
                                degrade_alpha=alpha,
                                degrade_target=degrade_target,
                            )
                            best_test_degrade[alpha] = (test_degrade_text, test_degrade_vis)
                            tag = _alpha_tag(alpha)
                            if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                                log_payload = {}
                                if test_degrade_text is not None:
                                    log_payload[f"Test_degrade_text_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_text)
                                if test_degrade_vis is not None:
                                    log_payload[f"Test_degrade_visual_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_vis)
                                if log_payload:
                                    wandb.log(log_payload)
                    elif degrade_vals is not None:
                        best_test_degrade = degrade_vals

            if args.early_stop_patience is not None:
                if stopper.step(val_score):
                    epochs_needed = epoch
                    break

            toc = time.time()
            total_time += toc - tic

            if epoch % args.log_every == 0:
                avg_epoch_time = float(total_time) / float(epoch)
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {avg_epoch_time:.2f}\n"
                    f"Loss: {_as_scalar_float(loss):.4f}\n"
                    f"Train/Val/Test loss: {_as_scalar_float(loss):.4f}/{_as_scalar_float(val_loss):.4f}/{_as_scalar_float(test_loss):.4f}\n"
                    f"Train/Val/Test/Best Val/Final Test {args.metric}: "
                    f"{_as_scalar_float(train_result):.4f}/{_as_scalar_float(val_result):.4f}/{_as_scalar_float(test_result):.4f}/"
                    f"{_as_scalar_float(best_val_result):.4f}/{_as_scalar_float(final_test_result):.4f}"
                )

    if is_cuda_run:
        peak_memory_mb = th.cuda.max_memory_allocated(model_device) / 1048576.0

    print("*" * 50)
    print(f"Best val  {args.metric}: {best_val_result}, Final test  {args.metric}: {final_test_result}")
    print("*" * 50)
    if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
        wandb.log({"summary/best_val_select": _as_scalar_float(best_val_score)})
    if report_drop and best_test_degrade is not None:
        if isinstance(best_test_degrade, dict):
            for alpha in degrade_alphas:
                if alpha not in best_test_degrade:
                    continue
                test_degrade_text, test_degrade_vis = best_test_degrade[alpha]
                tag = _alpha_tag(alpha)
                parts = []
                if test_degrade_text is not None:
                    parts.append(f"degrade-text {args.metric}: {_as_scalar_float(test_degrade_text):.4f}")
                if test_degrade_vis is not None:
                    parts.append(f"degrade-visual {args.metric}: {_as_scalar_float(test_degrade_vis):.4f}")
                if parts:
                    print(f"Best test degrade a{tag} | " + " | ".join(parts))
                if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                    log_payload = {}
                    if test_degrade_text is not None:
                        log_payload[f"summary/best_test_degrade_text_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_text)
                    if test_degrade_vis is not None:
                        log_payload[f"summary/best_test_degrade_visual_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_vis)
                    if log_payload:
                        wandb.log(log_payload)
        else:
            test_degrade_text, test_degrade_vis = best_test_degrade
            print(
                f"Best test degrade-text {args.metric}: {_as_scalar_float(test_degrade_text):.4f} | "
                f"degrade-visual {args.metric}: {_as_scalar_float(test_degrade_vis):.4f}"
            )
            if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                wandb.log(
                    {
                        f"summary/best_test_degrade_text_{args.metric}": _as_scalar_float(test_degrade_text),
                        f"summary/best_test_degrade_visual_{args.metric}": _as_scalar_float(test_degrade_vis),
                    }
                )

    if return_extra:
        # Compute avg epoch time
        if epochs_needed > 0:
            avg_epoch_time = float(total_time) / float(epochs_needed)
        else:
            avg_epoch_time = 0.0

        extra = {
            "best_val_select": _as_scalar_float(best_val_score),
            "best_val_metric": _as_scalar_float(best_val_result),
            "best_test_metric": _as_scalar_float(final_test_result),
            "best_state_dict": best_state_dict,
            "best_test_logits": best_test_logits,
            "peak_memory_mb": peak_memory_mb,
            "avg_epoch_time": avg_epoch_time,
            "epochs_needed": epochs_needed,
        }
        if report_drop and best_test_degrade is not None:
            if isinstance(best_test_degrade, dict):
                for alpha in degrade_alphas:
                    if alpha not in best_test_degrade:
                        continue
                    test_degrade_text, test_degrade_vis = best_test_degrade[alpha]
                    tag = _alpha_tag(alpha)
                    if test_degrade_text is not None:
                        extra[f"best_test_degrade_text_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_text)
                    if test_degrade_vis is not None:
                        extra[f"best_test_degrade_visual_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_vis)
                if len(degrade_alphas) == 1:
                    alpha = degrade_alphas[0]
                    test_degrade_text, test_degrade_vis = best_test_degrade.get(alpha, (None, None))
                    if test_degrade_text is not None:
                        extra[f"best_test_degrade_text_{args.metric}"] = _as_scalar_float(test_degrade_text)
                    if test_degrade_vis is not None:
                        extra[f"best_test_degrade_visual_{args.metric}"] = _as_scalar_float(test_degrade_vis)
            else:
                test_degrade_text, test_degrade_vis = best_test_degrade
                extra.update(
                    {
                        f"best_test_degrade_text_{args.metric}": _as_scalar_float(test_degrade_text),
                        f"best_test_degrade_visual_{args.metric}": _as_scalar_float(test_degrade_vis),
                    }
                )
        return best_val_result, final_test_result, extra

    return best_val_result, final_test_result


def main():
    parser = args_init()
    args = parser.parse_args()
    if str(getattr(args, "backend", "")).lower() == "mlp" and not str(getattr(args, "model_name", "")).strip():
        args.model_name = "MLP"

    if args.disable_wandb or wandb is None:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(config=args, reinit=True, entity="tiant-wang")

    if getattr(args, "result_csv", None) and not getattr(args, "report_drop_modality", False):
        args.report_drop_modality = True

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else "cpu")

    t0 = time.time()
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        name=args.data_name,
        fewshots=args.fewshots,
    )
    t_load_data = time.time()

    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    observe_graph = copy.deepcopy(graph)
    if args.inductive:
        observe_graph = _make_observe_graph_inductive(graph, val_idx, test_idx)

    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    t_graph_setup = time.time()

    text_feature = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    visual_feature = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
    t_numpy_load = time.time()
    n_classes = int((labels.max() + 1).item())
    print(
        f"Number of classes {n_classes}, "
        f"text dim {text_feature.shape[1]}, visual dim {visual_feature.shape[1]}"
    )

    graph.create_formats_()
    observe_graph.create_formats_()
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)
    observe_graph = observe_graph.to(device)
    t_tensor_transfer = time.time()

    print(f"[TIME] load_data: {t_load_data - t0:.1f}s | graph_setup: {t_graph_setup - t_load_data:.1f}s | numpy_load: {t_numpy_load - t_graph_setup:.1f}s | tensor_transfer: {t_tensor_transfer - t_numpy_load:.1f}s")

    proj_dim = int(args.mm_proj_dim) if args.mm_proj_dim is not None else int(args.n_hidden)
    early_fuse = str(getattr(args, "early_fuse", "concat")).lower().strip()
    if early_fuse not in ("concat", "sum"):
        raise ValueError(f"Unsupported --early_fuse: {early_fuse}")
    downstream_in_dim = proj_dim if early_fuse == "sum" else 2 * proj_dim
    single_modality = getattr(args, "single_modality", None)
    if single_modality in ("text", "visual"):
        downstream_in_dim = proj_dim

    val_results = []
    test_results = []
    degrade_text_results = []
    degrade_visual_results = []
    degrade_text_results_map = {}
    degrade_visual_results_map = {}

    # Efficiency profiling: collect per-run metrics
    efficiency_runs = {
        'peak_memory_MB': [],
        'epoch_times': [],
        'epochs_needed': [],
    }
    n_params_M = None  # will be set on first run

    t_train_start = time.time()
    for run in range(args.n_runs):
        set_seed(args.seed + run)

        separate_head = bool(getattr(args, "separate_classifier", False))
        embed_dim = int(args.early_embed_dim) if args.early_embed_dim is not None else int(args.n_hidden)
        use_no_encoder = bool(getattr(args, "early_no_encoder", False))

        if use_no_encoder:
            # No encoder: raw concat → GNN (traditional GNN approach)
            text_encoder = None
            visual_encoder = None
            if single_modality in ("text", "visual"):
                downstream_in_dim_gnn = int(text_feature.shape[1]) if single_modality == "text" else int(visual_feature.shape[1])
            else:
                downstream_in_dim_gnn = int(text_feature.shape[1]) + int(visual_feature.shape[1])
        else:
            text_encoder = ModalityEncoder(int(text_feature.shape[1]), proj_dim, args.dropout).to(device)
            visual_encoder = ModalityEncoder(int(visual_feature.shape[1]), proj_dim, args.dropout).to(device)
            if single_modality in ("text", "visual"):
                downstream_in_dim = proj_dim
            downstream_in_dim_gnn = downstream_in_dim

        if args.backend == "mlp":
            from GNN.Library.MLP import MLP

            out_dim = embed_dim if separate_head else n_classes
            mlp = MLP(downstream_in_dim_gnn, out_dim, args.n_layers, args.n_hidden, F.relu, args.dropout).to(device)
            base_model = SimpleMAGMLP(
                text_encoder,
                visual_encoder,
                mlp,
                early_fuse=early_fuse,
                single_modality=single_modality,
                use_no_encoder=use_no_encoder,
            )
        else:
            out_dim = embed_dim if separate_head else n_classes
            gnn = _build_gnn_backbone(args, downstream_in_dim_gnn, out_dim, device)
            base_model = SimpleMAGGNN(
                text_encoder,
                visual_encoder,
                gnn,
                early_fuse=early_fuse,
                single_modality=single_modality,
                use_mlp_projection=bool(getattr(args, "early_mlp_projection", False)),
                use_no_encoder=use_no_encoder,
            )

        if separate_head:
            head = nn.Linear(embed_dim, n_classes).to(device)
            model = _SeparateHeadWrapper(base_model, head)
            model.reset_parameters()
        else:
            model = base_model
            model.reset_parameters()

        # Count params on first run
        if n_params_M is None:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_params_M = n_params / 1e6

        if float(getattr(args, "mmcl_weight", 0.0)) > 0.0 or float(getattr(args, "kd_weight", 0.0)) > 0.0:
            val_result, test_result, extra = _mag_classification_mmcl(
                args,
                graph,
                observe_graph,
                model,
                text_feature,
                visual_feature,
                labels,
                train_idx,
                val_idx,
                test_idx,
                run + 1,
                return_extra=True,
            )
        else:
            val_result, test_result, extra = mag_classification(
                args,
                graph,
                observe_graph,
                model,
                text_feature,
                visual_feature,
                labels,
                train_idx,
                val_idx,
                test_idx,
                run + 1,
                return_extra=True,
            )

        if wandb is not None and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
            wandb.log({f"Val_{args.metric}": val_result, f"Test_{args.metric}": test_result})

        val_results.append(val_result)
        test_results.append(test_result)
        degrade_alphas = _parse_degrade_alphas(args)
        degrade_text_key = f"best_test_degrade_text_{args.metric}"
        degrade_visual_key = f"best_test_degrade_visual_{args.metric}"
        for alpha in degrade_alphas:
            tag = _alpha_tag(alpha)
            degrade_text_results_map[tag] = degrade_text_results_map.get(tag, [])
            degrade_visual_results_map[tag] = degrade_visual_results_map.get(tag, [])
            k_text = f"best_test_degrade_text_{args.metric}_a{tag}"
            k_vis = f"best_test_degrade_visual_{args.metric}_a{tag}"
            if extra.get(k_text) is not None:
                degrade_text_results_map[tag].append(extra[k_text])
            if extra.get(k_vis) is not None:
                degrade_visual_results_map[tag].append(extra[k_vis])
        if len(degrade_alphas) == 1:
            if extra.get(degrade_text_key) is not None:
                degrade_text_results.append(extra[degrade_text_key])
            if extra.get(degrade_visual_key) is not None:
                degrade_visual_results.append(extra[degrade_visual_key])

        best_state_dict = extra.get("best_state_dict") if isinstance(extra, dict) else None
        if best_state_dict is not None and getattr(args, "save_best_ckpt", None):
            ckpt_path = _build_run_ckpt_path(str(args.save_best_ckpt), run + 1)
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            th.save({"state_dict": best_state_dict, "args": vars(args)}, ckpt_path)
            print(f"[Info] Saved best checkpoint to {ckpt_path}")

        if best_state_dict is not None and getattr(args, "export_predictions", None):
            # Re-run forward pass with best model to get test predictions
            model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
            model.eval()
            with th.no_grad():
                best_pred = model(graph, text_feature, visual_feature)
                test_logits = best_pred[test_idx]
            pred_path = str(args.export_predictions)
            if args.n_runs > 1:
                root, ext = os.path.splitext(pred_path)
                pred_path = f"{root}_run{run+1}{ext}"
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            th.save(th.argmax(test_logits, dim=1), pred_path)
            print(f"[Export] Test predictions → {pred_path}")

        # Collect efficiency profiling data from extra
        if isinstance(extra, dict):
            efficiency_runs['peak_memory_MB'].append(extra.get('peak_memory_mb', 0.0))
            efficiency_runs['epochs_needed'].append(extra.get('epochs_needed', args.n_epochs))
            avg_ep_time = extra.get('avg_epoch_time', 0.0)
            ep_needed = extra.get('epochs_needed', args.n_epochs)
            efficiency_runs['epoch_times'].append([avg_ep_time] * ep_needed)

        # Clean up GPU memory between runs to prevent allocator cache accumulation
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

    def _mean_std(values):
        mean = float(np.mean(values))
        std = float(np.std(values))
        return mean, std

    def _fmt_pct(values):
        mean, std = _mean_std(values)
        return f"{mean * 100.0:.3f} ± {std * 100.0:.3f}%"

    print(f"Runned {args.n_runs} times")
    print(f"Average val {args.metric}: {_fmt_pct(val_results)}")
    print(f"Average test {args.metric}: {_fmt_pct(test_results)}")
    t_end = time.time()
    print(f"[TIME] total: {t_end - t0:.1f}s | setup: {t_train_start - t0:.1f}s | train({args.n_runs} runs): {t_end - t_train_start:.1f}s (excl. module import)")

    if wandb is not None and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
        val_mean, val_std = _mean_std(val_results)
        test_mean, test_std = _mean_std(test_results)
        wandb.log(
            {
                f"Mean_Val_{args.metric}": val_mean,
                f"Std_Val_{args.metric}": val_std,
                f"Mean_Test_{args.metric}": test_mean,
                f"Std_Test_{args.metric}": test_std,
            }
        )

    if getattr(args, "result_csv", None) or getattr(args, "result_csv_all", None):
        method = str(getattr(args, "result_tag", "") or "").strip()
        if not method:
            if str(getattr(args, "backend", "")).lower() == "mlp":
                method = "EarlyFusionMLP"
            else:
                method = "EarlyFusionGNN"
            fuse_tag = str(getattr(args, "early_fuse", "concat")).lower().strip()
            if fuse_tag and fuse_tag != "concat":
                method = f"{method}-{fuse_tag}"
            if float(getattr(args, "mmcl_weight", 0.0)) > 0.0:
                method = f"{method}+MMCL"

        test_mean = float(np.mean(test_results)) if test_results else None
        test_std = float(np.std(test_results)) if test_results else None
        degrade_text_mean = float(np.mean(degrade_text_results)) if degrade_text_results else None
        degrade_text_std = float(np.std(degrade_text_results)) if degrade_text_results else None
        degrade_visual_mean = float(np.mean(degrade_visual_results)) if degrade_visual_results else None
        degrade_visual_std = float(np.std(degrade_visual_results)) if degrade_visual_results else None
        key_fields = [
            "dataset",
            "method",
            "backbone",
            "n_layers",
            "single_modality",
            "inductive",
            "fewshots",
            "metric",
            "text_feature",
            "visual_feature",
        ]
        if bool(getattr(args, "best_ignore_layers", True)):
            key_fields = [k for k in key_fields if k != "n_layers"]

        per_alpha_extra = {}
        degrade_alphas = _parse_degrade_alphas(args)
        for alpha in degrade_alphas:
            tag = _alpha_tag(alpha)
            t_vals = degrade_text_results_map.get(tag, [])
            v_vals = degrade_visual_results_map.get(tag, [])
            if t_vals:
                t_mean = float(np.mean(t_vals))
                t_std = float(np.std(t_vals))
                per_alpha_extra[f"degrade_text_a{tag}"] = round(t_mean * 100.0, 3)
                per_alpha_extra[f"degrade_text_a{tag}_std"] = round(t_std * 100.0, 3)
                per_alpha_extra[f"degrade_text_a{tag}_pm"] = f"{t_mean * 100.0:.2f} ± {t_std * 100.0:.2f}"
            if v_vals:
                v_mean = float(np.mean(v_vals))
                v_std = float(np.std(v_vals))
                per_alpha_extra[f"degrade_visual_a{tag}"] = round(v_mean * 100.0, 3)
                per_alpha_extra[f"degrade_visual_a{tag}_std"] = round(v_std * 100.0, 3)
                per_alpha_extra[f"degrade_visual_a{tag}_pm"] = f"{v_mean * 100.0:.2f} ± {v_std * 100.0:.2f}"
        if test_mean is not None:
            row = build_result_row(
                args=args,
                method=method,
                full_metric=test_mean,
                degrade_text=degrade_text_mean,
                degrade_visual=degrade_visual_mean,
                extra={
                    "full_std": test_std,
                    "degrade_text_std": degrade_text_std,
                    "degrade_visual_std": degrade_visual_std,
                    **per_alpha_extra,
                },
            )
            if getattr(args, "result_csv", None):
                update_best_result_csv(args.result_csv, row, key_fields=key_fields, score_field="full")
            if getattr(args, "result_csv_all", None):
                append_result_csv(args.result_csv_all, row)

    # Efficiency profiling summary
    all_epoch_times = [t for run_times in efficiency_runs['epoch_times'] for t in run_times]
    avg_epoch_time = float(np.mean(all_epoch_times)) if all_epoch_times else 0
    std_epoch_time = float(np.std(all_epoch_times)) if all_epoch_times else 0
    avg_epochs_needed = float(np.mean(efficiency_runs['epochs_needed']))
    std_epochs_needed = float(np.std(efficiency_runs['epochs_needed']))
    avg_peak_memory = float(np.mean(efficiency_runs['peak_memory_MB']))
    std_peak_memory = float(np.std(efficiency_runs['peak_memory_MB']))
    avg_total_time = avg_epochs_needed * avg_epoch_time
    std_total_time = float(np.std([sum(run_times) for run_times in efficiency_runs['epoch_times']])) if efficiency_runs['epoch_times'] else 0

    print(f"\n{'='*60}")
    print(f"Efficiency Profile: Early_GNN on {args.data_name}")
    print(f"{'='*60}")
    print(f"  Parameters:       {n_params_M:.3f} M" if n_params_M else "  Parameters:       N/A")
    print(f"  Peak Memory:     {avg_peak_memory:.2f} ± {std_peak_memory:.2f} MB")
    print(f"  Total Time(est): {avg_total_time:.2f} ± {std_total_time:.2f} s  ({avg_total_time/60:.1f} min)")
    print(f"  Avg Epoch:        {avg_epoch_time:.4f} ± {std_epoch_time:.4f} s/epoch")
    print(f"  Epochs Needed:    {avg_epochs_needed:.1f} ± {std_epochs_needed:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

