import argparse
import copy
import gc
import os
import sys
import time
import re
from typing import Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

# Allow running as a script: `python GNN/Baselines/Late_GNN.py`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(_ROOT))

from GNN.GraphData import load_data, set_seed  # noqa: E402
from GNN.Utils.LossFunction import cross_entropy, get_metric  # noqa: E402
from GNN.Utils.model_config import str2bool
from GNN.Utils.NodeClassification import (  # noqa: E402
    initialize_early_stopping,
    initialize_optimizer_and_scheduler,
    adjust_learning_rate_if_needed,
    log_results_to_wandb,
    log_progress,
    print_final_results,
    _as_scalar_float,
)
from GNN.Utils.model_config import (  # noqa: E402
    add_common_args,
    add_sage_args,
    add_gat_args,
    add_revgat_args,
    add_sgc_args,
    add_appnp_args,
)
from GNN.Utils.result_logger import build_result_row, update_best_result_csv, append_result_csv  # noqa: E402
import GNN.Baselines.Early_GNN as mag_base  # noqa: E402


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


def _make_noisy_feature(feat: th.Tensor, train_idx: th.Tensor, alpha: float) -> th.Tensor:
    if alpha <= 0.0:
        return feat
    base_idx = th.arange(feat.shape[0], device=feat.device)
    mean_feat = feat[base_idx].mean(dim=0, keepdim=True)
    std_feat = feat[base_idx].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    noise = th.randn_like(feat) * std_feat + mean_feat
    if alpha >= 1.0:
        return noise
    return feat * (1.0 - alpha) + noise * alpha


def _parse_degrade_alphas(args) -> list:
    raw = getattr(args, "degrade_alphas", "")
    if raw is None or str(raw).strip() == "":
        return [float(getattr(args, "degrade_alpha", 1.0))]
    parts = re.split(r"[\s,]+", str(raw).strip())
    alphas = []
    for p in parts:
        if not p:
            continue
        alphas.append(float(p))
    return alphas


def _alpha_tag(alpha: float) -> str:
    return f"{int(round(float(alpha) * 100))}"


class LateFusionMAG(nn.Module):
    """Per-modality GNN branches with representation-level late fusion."""

    def __init__(
        self,
        text_encoder: Optional[nn.Module],
        visual_encoder: Optional[nn.Module],
        text_gnn: nn.Module,
        visual_gnn: nn.Module,
        classifier: nn.Module,
        use_mlp_before_fusion: bool = False,
        use_no_encoder: bool = False,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.text_gnn = text_gnn
        self.visual_gnn = visual_gnn
        self.classifier = classifier
        self.use_mlp_before_fusion = use_mlp_before_fusion
        self.use_no_encoder = use_no_encoder
        if use_mlp_before_fusion and not use_no_encoder:
            enc_dim = text_encoder.proj.out_features
            self.text_mlp = nn.Sequential(
                nn.Linear(enc_dim, enc_dim),
                nn.ReLU(),
                nn.LayerNorm(enc_dim),
                nn.Linear(enc_dim, enc_dim),
            )
            self.vis_mlp = nn.Sequential(
                nn.Linear(enc_dim, enc_dim),
                nn.ReLU(),
                nn.LayerNorm(enc_dim),
                nn.Linear(enc_dim, enc_dim),
            )

    def reset_parameters(self):
        if self.text_encoder is not None and hasattr(self.text_encoder, "reset_parameters"):
            self.text_encoder.reset_parameters()
        if self.visual_encoder is not None and hasattr(self.visual_encoder, "reset_parameters"):
            self.visual_encoder.reset_parameters()
        if hasattr(self.text_gnn, "reset_parameters"):
            self.text_gnn.reset_parameters()
        if hasattr(self.visual_gnn, "reset_parameters"):
            self.visual_gnn.reset_parameters()
        if hasattr(self.classifier, "reset_parameters"):
            self.classifier.reset_parameters()

    def fuse_embeddings(self, text_h: th.Tensor, vis_h: th.Tensor) -> th.Tensor:
        # Concat fusion for fair comparison with Early Fusion baselines
        return th.cat([text_h, vis_h], dim=1)

    def forward(self, graph, text_feature: th.Tensor, visual_feature: th.Tensor) -> th.Tensor:
        text_h, vis_h = self.forward_branches(graph, text_feature, visual_feature)
        fused = self.fuse_embeddings(text_h, vis_h)
        return self.classifier(fused)

    def forward_branches(self, graph, text_feature: th.Tensor, visual_feature: th.Tensor):
        if self.use_no_encoder:
            # No encoder: raw features → GNN directly
            text_z = text_feature
            vis_z = visual_feature
        else:
            text_z = self.text_encoder(text_feature)
            vis_z = self.visual_encoder(visual_feature)
            if self.use_mlp_before_fusion:
                text_z = self.text_mlp(text_z)
                vis_z = self.vis_mlp(vis_z)
        # GCNII uses forward(x, adj) signature - swap args for GCNII only
        if type(self.text_gnn).__name__ == "GCNII":
            text_h = self.text_gnn(text_z, graph)
            vis_h = self.visual_gnn(vis_z, graph)
        else:
            text_h = self.text_gnn(graph, text_z)
            vis_h = self.visual_gnn(graph, vis_z)
        return text_h, vis_h


def _load_mag_context(args, device: th.device):
    graph, labels, train_idx, val_idx, test_idx = load_data(
        args.graph_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        name=args.data_name,
        fewshots=args.fewshots,
    )

    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    observe_graph = copy.deepcopy(graph)
    if args.inductive:
        observe_graph = mag_base._make_observe_graph_inductive(graph, val_idx, test_idx)

    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    graph.create_formats_()
    observe_graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)
    observe_graph = observe_graph.to(device)

    text_feature = th.from_numpy(np.load(args.text_feature).astype(np.float32)).to(device)
    visual_feature = th.from_numpy(np.load(args.visual_feature).astype(np.float32)).to(device)
    n_classes = int((labels.max() + 1).item())

    return graph, observe_graph, labels, train_idx, val_idx, test_idx, text_feature, visual_feature, n_classes


def args_init():
    parser = argparse.ArgumentParser(
        "MAG Late Fusion (Modality Separation: per-modality GNNs + fuse)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    add_sage_args(parser)
    add_gat_args(parser)
    add_revgat_args(parser)
    add_sgc_args(parser)
    add_appnp_args(parser)

    parser.add_argument("--gcnii_lamda", type=float, default=0.5, help="GCNII lamda (initial residual strength)")
    parser.add_argument("--gcnii_alpha", type=float, default=0.5, help="GCNII alpha (residual coefficient)")
    parser.add_argument("--gcnii_variant", action="store_true", help="GCNII variant (concatenate initial features)")
    parser.add_argument("--jknet_aggr", type=str, default="last", choices=["concat", "max", "last"], help="JKNet aggregation mode")

    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging for offline runs")
    parser.add_argument(
        "--late_embed_dim",
        type=int,
        default=None,
        help="Embedding dim for late fusion (default: n_hidden).",
    )
    parser.add_argument("--mm_proj_dim", type=int, default=None, help="Per-modality encoder output dim")
    parser.add_argument("--mmcl_weight", type=float, default=0.0, help="Weight for modality contrastive loss")
    parser.add_argument("--mmcl_temperature", type=float, default=0.1, help="Temperature for modality contrastive loss")
    parser.add_argument("--mmcl_max_pairs", type=int, default=2048, help="Max train nodes for contrastive loss")
    parser.add_argument(
        "--degrade_alpha",
        type=float,
        default=1.0,
        help="Noise strength for modality degradation at eval (0=no degrade, 1=full noise).",
    )
    parser.add_argument(
        "--save_best_ckpt",
        type=str,
        default=None,
        help="Optional path to save best checkpoint per run (weights-only + args). If multiple runs, appends _run{k}.pt.",
    )
    parser.add_argument("--export_predictions", type=str, default=None,
                        help="Path to save test predictions as torch.Tensor (argmax, shape=[N_test])")
    parser.add_argument(
        "--late_mlp_before_fusion",
        type=str2bool,
        default=False,
        help="Add MLP projection before late fusion concatenation.",
    )
    parser.add_argument(
        "--late_no_encoder",
        type=str2bool,
        default=True,
        help="Skip per-modality encoders and pass raw features directly to GNN (default: True).",
    )
    parser.add_argument("--analyze_gradients", action="store_true",
                        help="Enable per-epoch gradient L2 norm tracking")
    parser.add_argument("--gradient_csv", type=str, default=None,
                        help="Path to save gradient L2 norm CSV")
    return parser


def _build_run_ckpt_path(base_path: str, run_idx: int) -> str:
    base_path = str(base_path)
    root, ext = os.path.splitext(base_path)
    if ext.lower() != ".pt":
        ext = ".pt"
    if run_idx <= 1:
        return f"{root}{ext}"
    return f"{root}_run{run_idx}{ext}"


def main():
    parser = args_init()
    args = parser.parse_args()

    if args.disable_wandb or wandb is None:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(config=args, reinit=True, entity="tiant-wang")

    if getattr(args, "result_csv", None) and not getattr(args, "report_drop_modality", False):
        args.report_drop_modality = True

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else "cpu")

    t0 = time.time()
    (
        graph,
        observe_graph,
        labels,
        train_idx,
        val_idx,
        test_idx,
        text_feat,
        vis_feat,
        n_classes,
    ) = _load_mag_context(args, device)
    t_load_data = time.time()

    val_results = []
    test_results = []
    degrade_text_results = []
    degrade_visual_results = []

    select_metric = args.metric
    select_average = args.average
    report_drop = bool(getattr(args, "report_drop_modality", False))
    report_drop_mode = str(getattr(args, "report_drop_mode", "best")).lower()
    degrade_target = str(getattr(args, "degrade_target", "both")).lower()
    degrade_alphas = _parse_degrade_alphas(args)
    degrade_text_results_map = {}
    degrade_visual_results_map = {}

    # Efficiency profiling: collect per-run metrics
    efficiency_runs = {
        'peak_memory_MB': [],
        'epoch_times': [],
        'epochs_needed': [],
    }

    # Build one model to count params (before the run loop)
    _proj_dim = int(args.mm_proj_dim) if args.mm_proj_dim is not None else int(args.n_hidden)
    _embed_dim = int(args.late_embed_dim) if args.late_embed_dim is not None else int(args.n_hidden)
    _use_no_encoder = bool(getattr(args, "late_no_encoder", False))
    if _use_no_encoder:
        _text_encoder = None
        _visual_encoder = None
        _text_gnn = mag_base._build_gnn_backbone(args, int(text_feat.shape[1]), _embed_dim, device)
        _vis_gnn = mag_base._build_gnn_backbone(args, int(vis_feat.shape[1]), _embed_dim, device)
    else:
        _text_encoder = mag_base.ModalityEncoder(int(text_feat.shape[1]), _proj_dim, float(args.dropout)).to(device)
        _visual_encoder = mag_base.ModalityEncoder(int(vis_feat.shape[1]), _proj_dim, float(args.dropout)).to(device)
        _text_gnn = mag_base._build_gnn_backbone(args, _proj_dim, _embed_dim, device)
        _vis_gnn = mag_base._build_gnn_backbone(args, _proj_dim, _embed_dim, device)
    _classifier = nn.Linear(2 * _embed_dim, n_classes).to(device)
    _model_for_count = LateFusionMAG(
        _text_encoder, _visual_encoder, _text_gnn, _vis_gnn, _classifier,
        use_mlp_before_fusion=bool(getattr(args, "late_mlp_before_fusion", False)),
        use_no_encoder=_use_no_encoder,
    )
    n_params = sum(p.numel() for p in _model_for_count.parameters() if p.requires_grad)
    n_params_M = n_params / 1e6
    del _model_for_count

    for run in range(args.n_runs):
        set_seed(args.seed + run)

        proj_dim = int(args.mm_proj_dim) if args.mm_proj_dim is not None else int(args.n_hidden)
        embed_dim = int(args.late_embed_dim) if args.late_embed_dim is not None else int(args.n_hidden)
        use_no_encoder = bool(getattr(args, "late_no_encoder", False))
        if use_no_encoder:
            text_encoder = None
            visual_encoder = None
            # Build GNN accepting raw feature dims
            text_gnn = mag_base._build_gnn_backbone(args, int(text_feat.shape[1]), embed_dim, device)
            vis_gnn = mag_base._build_gnn_backbone(args, int(vis_feat.shape[1]), embed_dim, device)
        else:
            text_encoder = mag_base.ModalityEncoder(int(text_feat.shape[1]), proj_dim, float(args.dropout)).to(device)
            visual_encoder = mag_base.ModalityEncoder(int(vis_feat.shape[1]), proj_dim, float(args.dropout)).to(device)
            text_gnn = mag_base._build_gnn_backbone(args, proj_dim, embed_dim, device)
            vis_gnn = mag_base._build_gnn_backbone(args, proj_dim, embed_dim, device)
        classifier = nn.Linear(2 * embed_dim, n_classes).to(device)
        model = LateFusionMAG(
            text_encoder,
            visual_encoder,
            text_gnn,
            vis_gnn,
            classifier,
            use_mlp_before_fusion=bool(getattr(args, "late_mlp_before_fusion", False)),
            use_no_encoder=use_no_encoder,
        )
        model.reset_parameters()

        stopper = initialize_early_stopping(args)
        optimizer, lr_scheduler = initialize_optimizer_and_scheduler(args, model)

        # Per-epoch gradient L2 norm history (for gradient starvation verification)
        grad_history = {'text_gnn': [], 'vis_gnn': [], 'mmgnn': []}

        total_time = 0
        best_val_result, final_test_result = -1.0, 0.0
        best_val_score = -1.0
        best_test_degrade = None
        best_state_dict = None

        # Efficiency tracking
        peak_memory_mb = 0.0
        if th.cuda.is_available():
            th.cuda.reset_peak_memory_stats(device)
            th.cuda.empty_cache()
        epochs_needed = args.n_epochs  # will be updated if early stop triggered

        for epoch in range(1, args.n_epochs + 1):
            tic = time.time()
            adjust_learning_rate_if_needed(args, optimizer, epoch)

            model.train()
            optimizer.zero_grad()
            text_h, vis_h = model.forward_branches(observe_graph, text_feat, vis_feat)
            fused = model.fuse_embeddings(text_h, vis_h)
            logits = model.classifier(fused)
            train_loss = cross_entropy(logits[train_idx], labels[train_idx], label_smoothing=args.label_smoothing)
            con_loss = th.tensor(0.0, device=train_loss.device)
            mmcl_weight = float(getattr(args, "mmcl_weight", 0.0))
            kd_weight = float(getattr(args, "kd_weight", 0.0))
            kd_temperature = float(getattr(args, "kd_temperature", 1.0))
            if mmcl_weight > 0.0:
                # Align with EarlyFusion: apply CL on modality-encoder outputs (before any GNN propagation).
                text_z = model.text_encoder(text_feat)
                vis_z = model.visual_encoder(vis_feat)
                pair_idx = _sample_pair_idx(train_idx, int(getattr(args, "mmcl_max_pairs", 2048)))
                if int(pair_idx.numel()) > 1:
                    con_loss = _info_nce(
                        text_z[pair_idx],
                        vis_z[pair_idx],
                        temperature=float(getattr(args, "mmcl_temperature", 0.1)),
                    )
            kd_loss = th.tensor(0.0, device=train_loss.device)
            if kd_weight > 0.0:
                teacher_logits = logits.detach()
                logits_text = model.classifier(text_h)
                logits_vis = model.classifier(vis_h)
                kd_loss = 0.5 * (
                    _kd_loss(teacher_logits[train_idx], logits_text[train_idx], temperature=kd_temperature)
                    + _kd_loss(teacher_logits[train_idx], logits_vis[train_idx], temperature=kd_temperature)
                )
            total_loss = train_loss + mmcl_weight * con_loss + kd_weight * kd_loss
            total_loss.backward()

            # Record gradient L2 norms for gradient starvation verification
            # MMGCN: text/visual features → concat → modality-specific GNN
            # Encoders (text_encoder/visual_encoder) may not exist when late_no_encoder=True
            # So we always track the GNN layers: text_gnn and vis_gnn
            if getattr(args, 'analyze_gradients', False):
                def _grad_norm_sq(m):
                    return sum(
                        p.grad.float().norm(2).pow(2).item()
                        for p in m.parameters()
                        if p.grad is not None
                    )
                grad_history['text_gnn'].append(_grad_norm_sq(model.text_gnn) ** 0.5)
                grad_history['vis_gnn'].append(_grad_norm_sq(model.visual_gnn) ** 0.5)
                mmgnn_sq = _grad_norm_sq(model.text_gnn) + _grad_norm_sq(model.visual_gnn)
                grad_history['mmgnn'].append(mmgnn_sq ** 0.5)

            optimizer.step()

            if epoch % args.eval_steps == 0:
                model.eval()
                degrade_vals = None
                with th.no_grad():
                    text_h, vis_h = model.forward_branches(graph, text_feat, vis_feat)
                    fused = model.fuse_embeddings(text_h, vis_h)
                    logits_e = model.classifier(fused)
                    if report_drop and report_drop_mode == "always":
                        degrade_vals = {}
                        for alpha in degrade_alphas:
                            do_text = degrade_target in ("text", "both")
                            do_visual = degrade_target in ("visual", "both")
                            test_degrade_text = None
                            test_degrade_vis = None
                            if do_text:
                                noisy_text = _make_noisy_feature(text_feat, train_idx, float(alpha))
                                logits_degrade_text = model(graph, noisy_text, vis_feat)
                                test_degrade_text = get_metric(
                                    th.argmax(logits_degrade_text[test_idx], dim=1), labels[test_idx], args.metric, average=args.average
                                )
                            if do_visual:
                                noisy_vis = _make_noisy_feature(vis_feat, train_idx, float(alpha))
                                logits_degrade_vis = model(graph, text_feat, noisy_vis)
                                test_degrade_vis = get_metric(
                                    th.argmax(logits_degrade_vis[test_idx], dim=1), labels[test_idx], args.metric, average=args.average
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

                val_loss = cross_entropy(logits_e[val_idx], labels[val_idx], label_smoothing=args.label_smoothing)
                test_loss = cross_entropy(logits_e[test_idx], labels[test_idx], label_smoothing=args.label_smoothing)

                train_result = get_metric(th.argmax(logits_e[train_idx], dim=1), labels[train_idx], args.metric, average=args.average)
                val_result = get_metric(th.argmax(logits_e[val_idx], dim=1), labels[val_idx], args.metric, average=args.average)
                test_result = get_metric(th.argmax(logits_e[test_idx], dim=1), labels[test_idx], args.metric, average=args.average)

                log_results_to_wandb(total_loss, val_loss, test_loss, train_result, val_result, test_result)
                if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                    wandb.log(
                        {
                            "Train_contrastive_loss": _as_scalar_float(con_loss),
                            "Train_kd_loss": _as_scalar_float(kd_loss),
                        }
                    )
                lr_scheduler.step(_as_scalar_float(total_loss))

                val_pred = th.argmax(logits_e[val_idx], dim=1)
                val_true = labels[val_idx]
                val_score = get_metric(val_pred, val_true, select_metric, average=select_average)
                if report_drop and degrade_vals is not None and (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                    test_degrade_text, test_degrade_vis = degrade_vals
                    wandb.log({
                        f"Test_degrade_text_{args.metric}": _as_scalar_float(test_degrade_text),
                        f"Test_degrade_visual_{args.metric}": _as_scalar_float(test_degrade_vis),
                    })

                if val_score > best_val_score:
                    best_val_score = float(val_score)
                    best_val_result = float(val_result)
                    final_test_result = float(test_result)
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if report_drop:
                        if report_drop_mode == "best":
                            best_test_degrade = {}
                            for alpha in degrade_alphas:
                                do_text = degrade_target in ("text", "both")
                                do_visual = degrade_target in ("visual", "both")
                                test_degrade_text = None
                                test_degrade_vis = None
                                if do_text:
                                    noisy_text = _make_noisy_feature(text_feat, train_idx, float(alpha))
                                    logits_degrade_text = model(graph, noisy_text, vis_feat)
                                    test_degrade_text = get_metric(
                                        th.argmax(logits_degrade_text[test_idx], dim=1), labels[test_idx], args.metric, average=args.average
                                    )
                                if do_visual:
                                    noisy_vis = _make_noisy_feature(vis_feat, train_idx, float(alpha))
                                    logits_degrade_vis = model(graph, text_feat, noisy_vis)
                                    test_degrade_vis = get_metric(
                                        th.argmax(logits_degrade_vis[test_idx], dim=1), labels[test_idx], args.metric, average=args.average
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

                if stopper and stopper.step(val_score):
                    epochs_needed = epoch
                    break

                log_progress(
                    args,
                    epoch,
                    run + 1,
                    total_time,
                    total_loss,
                    val_loss,
                    test_loss,
                    train_result,
                    val_result,
                    test_result,
                    best_val_result,
                    final_test_result,
                )

            total_time += time.time() - tic

        print_final_results(best_val_result, final_test_result, args)

        if best_state_dict is not None and getattr(args, "save_best_ckpt", None):
            ckpt_path = _build_run_ckpt_path(str(args.save_best_ckpt), run + 1)
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            th.save({"state_dict": best_state_dict, "args": vars(args)}, ckpt_path)
            print(f"[Info] Saved best checkpoint to {ckpt_path}")
        if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
            wandb.log({"summary/best_val_select": _as_scalar_float(best_val_score)})
        if report_drop and best_test_degrade is not None:
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

        val_results.append(best_val_result)
        test_results.append(final_test_result)
        if getattr(args, 'export_predictions', None) and best_state_dict is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
            model.eval()
            with th.no_grad():
                text_h, vis_h = model.forward_branches(graph, text_feat, vis_feat)
                fused = model.fuse_embeddings(text_h, vis_h)
                logits_e = model.classifier(fused)
                test_logits = logits_e[test_idx]
            pred_path = str(args.export_predictions)
            if args.n_runs > 1:
                root, ext = os.path.splitext(pred_path)
                pred_path = f"{root}_run{run+1}{ext}"
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            th.save(th.argmax(test_logits, dim=1), pred_path)
            print(f"[Export] Test predictions → {pred_path}")
        if best_test_degrade is not None:
            for alpha in degrade_alphas:
                if alpha not in best_test_degrade:
                    continue
                degrade_text, degrade_vis = best_test_degrade[alpha]
                tag = _alpha_tag(alpha)
                degrade_text_results_map.setdefault(tag, [])
                degrade_visual_results_map.setdefault(tag, [])
                if degrade_text is not None:
                    degrade_text_results_map[tag].append(degrade_text)
                if degrade_vis is not None:
                    degrade_visual_results_map[tag].append(degrade_vis)
            if len(degrade_alphas) == 1:
                alpha = degrade_alphas[0]
                if alpha in best_test_degrade:
                    degrade_text, degrade_vis = best_test_degrade[alpha]
                    if degrade_text is not None:
                        degrade_text_results.append(degrade_text)
                    if degrade_vis is not None:
                        degrade_visual_results.append(degrade_vis)

        # Collect efficiency profiling data
        if th.cuda.is_available():
            th.cuda.synchronize()
            peak_memory_mb = th.cuda.max_memory_allocated(device) / 1048576.0
        avg_epoch_time = float(total_time) / float(epochs_needed) if epochs_needed > 0 else 0.0
        efficiency_runs['peak_memory_MB'].append(peak_memory_mb)
        efficiency_runs['epoch_times'].append([avg_epoch_time] * epochs_needed)
        efficiency_runs['epochs_needed'].append(epochs_needed)

        # Save per-epoch gradient L2 norm for this run (Group 1: MMGCN)
        if getattr(args, 'analyze_gradients', False) and getattr(args, 'gradient_csv', None):
            import csv
            grad_csv_path = args.gradient_csv.replace('.csv', f'_l2_norm_run{run+1}.csv')
            os.makedirs(os.path.dirname(grad_csv_path), exist_ok=True)
            with open(grad_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'text_gnn', 'vis_gnn', 'mmgnn'])
                for epoch_idx, (te, ve, mg) in enumerate(
                    zip(grad_history['text_gnn'], grad_history['vis_gnn'], grad_history['mmgnn']), start=1
                ):
                    writer.writerow([epoch_idx, te, ve, mg])
            print(f"[Run {run+1}] Gradient L2 norm saved to {grad_csv_path}")

        # Clean up GPU memory between runs to prevent allocator cache accumulation
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

        # metric-only reporting

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
    print(f"[TIME] total: {t_end - t0:.1f}s | setup: {t_load_data - t0:.1f}s | train: {t_end - t_load_data:.1f}s (excl. module import)")

    # Efficiency profiling summary
    all_epoch_times = [t for run_times in efficiency_runs['epoch_times'] for t in run_times]
    avg_epoch_time = float(np.mean(all_epoch_times)) if all_epoch_times else 0
    std_epoch_time = float(np.std(all_epoch_times)) if all_epoch_times else 0
    avg_epochs_needed = float(np.mean(efficiency_runs['epochs_needed']))
    std_epochs_needed = float(np.std(efficiency_runs['epochs_needed']))
    avg_peak_memory = float(np.mean(efficiency_runs['peak_memory_MB']))
    std_peak_memory = float(np.std(efficiency_runs['peak_memory_MB']))
    avg_total_time = avg_epochs_needed * avg_epoch_time
    std_total_time = float(np.std([sum(run_times) for run_times in efficiency_runs['epoch_times']]))

    print(f"\n{'='*60}")
    print(f"Efficiency Profile: Late_GNN on {args.data_name}")
    print(f"{'='*60}")
    print(f"  Parameters:       {n_params_M:.3f} M")
    print(f"  Peak Memory:     {avg_peak_memory:.2f} ± {std_peak_memory:.2f} MB")
    print(f"  Total Time(est): {avg_total_time:.2f} ± {std_total_time:.2f} s  ({avg_total_time/60:.1f} min)")
    print(f"  Avg Epoch:        {avg_epoch_time:.4f} ± {std_epoch_time:.4f} s/epoch")
    print(f"  Epochs Needed:    {avg_epochs_needed:.1f} ± {std_epochs_needed:.1f}")
    print(f"{'='*60}")

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
        method = str(getattr(args, "result_tag", "") or "").strip() or "MMGNN"
        if not str(getattr(args, "result_tag", "") or "").strip() and float(getattr(args, "mmcl_weight", 0.0)) > 0.0:
            method = f"{method}+MMCL"
        test_mean = float(np.mean(test_results)) if test_results else None
        test_std = float(np.std(test_results)) if test_results else None
        degrade_text_mean = float(np.mean(degrade_text_results)) if degrade_text_results else None
        degrade_text_std = float(np.std(degrade_text_results)) if degrade_text_results else None
        degrade_visual_mean = float(np.mean(degrade_visual_results)) if degrade_visual_results else None
        degrade_visual_std = float(np.std(degrade_visual_results)) if degrade_visual_results else None
        per_alpha_extra = {}
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
            if getattr(args, "result_csv", None):
                update_best_result_csv(args.result_csv, row, key_fields=key_fields, score_field="full")
            if getattr(args, "result_csv_all", None):
                append_result_csv(args.result_csv_all, row)


if __name__ == "__main__":
    main()

