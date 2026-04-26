import time
import os
import re
import torch as th
import torch.optim as optim
from GNN.Utils.LossFunction import cross_entropy, get_metric, EarlyStopping, adjust_learning_rate

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


def _as_scalar_float(value):
    if isinstance(value, th.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar tensor, got shape={tuple(value.shape)}")
        return float(value.detach().item())
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    return float(value)

def _normalize_average(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return value


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




def _compute_acc_f1(pred, labels, idx):
    pred_labels = th.argmax(pred[idx], dim=1)
    true_labels = labels[idx]
    acc = get_metric(pred_labels, true_labels, "accuracy")
    f1_macro = get_metric(pred_labels, true_labels, "f1", average="macro")
    return acc, f1_macro


def _compute_metric_from_logits(pred, labels, idx, metric, average=None):
    pred_labels = th.argmax(pred[idx], dim=1)
    true_labels = labels[idx]
    return get_metric(pred_labels, true_labels, metric, average=average)


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


@th.no_grad()
def _compute_degrade_metrics_mag(
    model,
    graph,
    text_feat,
    visual_feat,
    labels,
    idx,
    metric,
    average=None,
    train_idx=None,
    degrade_alpha: float = 1.0,
    degrade_target: str = "both",
    **model_kwargs,
):
    """Compute metric when degrading text or visual modality with Gaussian noise.

    Args:
        model: forward function that takes (graph, text_feat, visual_feat, **model_kwargs)
        **model_kwargs: extra arguments passed to model (e.g., pre-computed SIGN features)
    """
    target = str(degrade_target or "both").lower()
    do_text = target in ("text", "both")
    do_visual = target in ("visual", "both")

    degrade_text = None
    degrade_vis = None
    if do_text:
        noisy_text = _make_noisy_feature(text_feat, train_idx, float(degrade_alpha))
        pred_degrade_text = model(graph, noisy_text, visual_feat, **model_kwargs)
        degrade_text = _compute_metric_from_logits(pred_degrade_text, labels, idx, metric, average=average)
    if do_visual:
        noisy_vis = _make_noisy_feature(visual_feat, train_idx, float(degrade_alpha))
        pred_degrade_vis = model(graph, text_feat, noisy_vis, **model_kwargs)
        degrade_vis = _compute_metric_from_logits(pred_degrade_vis, labels, idx, metric, average=average)
    return degrade_text, degrade_vis


def train(model, graph, feat, labels, train_idx, optimizer, label_smoothing):
    model.train()

    optimizer.zero_grad()
    pred = model(graph, feat)
    loss = cross_entropy(pred[train_idx], labels[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(
        model, graph, feat, labels, train_idx, val_idx, test_idx, metric='accuracy', label_smoothing=0.1, average=None,
        return_pred=False
):
    model.eval()
    with th.no_grad():
        pred = model(graph, feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx], label_smoothing)
    test_loss = cross_entropy(pred[test_idx], labels[test_idx], label_smoothing)

    train_results = get_metric(th.argmax(pred[train_idx], dim=1), labels[train_idx], metric, average=average)
    val_results = get_metric(th.argmax(pred[val_idx], dim=1), labels[val_idx], metric, average=average)
    test_results = get_metric(th.argmax(pred[test_idx], dim=1), labels[test_idx], metric, average=average)

    if return_pred:
        return train_results, val_results, test_results, val_loss, test_loss, pred
    return train_results, val_results, test_results, val_loss, test_loss
def classification(
    args,
    graph,
    observe_graph,
    model,
    feat,
    labels,
    train_idx,
    val_idx,
    test_idx,
    n_running,
    save_path=None,
    return_pred=False,
):
    stopper = initialize_early_stopping(args)
    optimizer, lr_scheduler = initialize_optimizer_and_scheduler(args, model)

    total_time = 0
    best_val_result, final_test_result, best_val_loss = -1.0, 0, float("inf")
    best_val_score = -1.0
    select_metric = getattr(args, "metric", "accuracy")
    select_average = _normalize_average(getattr(args, "average", None))
    best_state_dict = None

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        adjust_learning_rate_if_needed(args, optimizer, epoch)

        train_loss, pred = train_model(model, observe_graph, feat, labels, train_idx, optimizer, args)

        if epoch % args.eval_steps == 0:
            train_result, val_result, test_result, val_loss, test_loss, pred = evaluate_model(
                args, model, graph, feat, labels, train_idx, val_idx, test_idx, return_pred=True
            )
            log_results_to_wandb(train_loss, val_loss, test_loss, train_result, val_result, test_result)
            lr_scheduler.step(_as_scalar_float(train_loss))

            total_time += time.time() - tic

            val_pred = th.argmax(pred[val_idx], dim=1)
            val_true = labels[val_idx]
            val_score = get_metric(val_pred, val_true, select_metric, average=select_average)
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_result = val_result
                final_test_result = test_result
                # Keep a CPU copy of weights to avoid GPU memory growth.
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if save_path is not None:
                    os.makedirs(save_path, exist_ok=True)
                    # Save weights-only to avoid pickle / torch.load(weights_only) issues.
                    th.save(best_state_dict, os.path.join(save_path, "model_state.pt"))

            if should_early_stop(stopper, val_score):
                break

            log_progress(args, epoch, n_running, total_time, train_loss, val_loss, test_loss, train_result, val_result, test_result, best_val_result, final_test_result)

    print_final_results(best_val_result, final_test_result, args)
    if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
        wandb.log({"summary/best_val_select": _as_scalar_float(best_val_score)})

    if save_path is not None or return_pred:
        if best_state_dict is None:
            best_state_dict = model.state_dict()
        model.load_state_dict(best_state_dict)
        model.eval()
        with th.no_grad():
            pred = model(graph, feat)
        return best_val_result, final_test_result, pred

    return best_val_result, final_test_result


def initialize_early_stopping(args):
    return EarlyStopping(patience=args.early_stop_patience) if args.early_stop_patience else None


def initialize_optimizer_and_scheduler(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    try:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=args.min_lr
        )
    except TypeError:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=100, min_lr=args.min_lr
        )
    return optimizer, lr_scheduler


def adjust_learning_rate_if_needed(args, optimizer, epoch):
    if args.warmup_epochs:
        adjust_learning_rate(optimizer, args.lr, epoch, args.warmup_epochs)


def train_model(model, observe_graph, feat, labels, train_idx, optimizer, args):
    return train(model, observe_graph, feat, labels, train_idx, optimizer, label_smoothing=args.label_smoothing)


def evaluate_model(args, model, graph, feat, labels, train_idx, val_idx, test_idx, return_pred=False):
    return evaluate(
        model, graph, feat, labels, train_idx, val_idx, test_idx, args.metric, args.label_smoothing, args.average,
        return_pred=return_pred
    )


def log_results_to_wandb(train_loss, val_loss, test_loss, train_result, val_result, test_result):
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        return
    if wandb is None:
        return
    wandb.log({
        'Train_loss': _as_scalar_float(train_loss),
        'Val_loss': _as_scalar_float(val_loss),
        'Test_loss': _as_scalar_float(test_loss),
        'Train_result': _as_scalar_float(train_result),
        'Val_result': _as_scalar_float(val_result),
        'Test_result': _as_scalar_float(test_result),
    })


def update_best_results(val_result, test_result, save_path, model):
    """Deprecated: kept for backward compatibility within this module."""
    best_val_result = val_result
    final_test_result = test_result
    return best_val_result, final_test_result


def should_early_stop(stopper, val_result):
    return stopper and stopper.step(val_result)


def log_progress(args, epoch, n_running, total_time, train_loss, val_loss, test_loss, train_result, val_result, test_result, best_val_result, final_test_result):
    if epoch % args.log_every == 0:
        avg_epoch_time = float(total_time) / float(epoch)
        train_loss = _as_scalar_float(train_loss)
        val_loss = _as_scalar_float(val_loss)
        test_loss = _as_scalar_float(test_loss)
        train_result = _as_scalar_float(train_result)
        val_result = _as_scalar_float(val_result)
        test_result = _as_scalar_float(test_result)
        best_val_result = _as_scalar_float(best_val_result)
        final_test_result = _as_scalar_float(final_test_result)
        print(
            f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {avg_epoch_time:.2f}\n"
            f"Loss: {train_loss:.4f}\n"
            f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
            f"Train/Val/Test/Best Val/Final Test {args.metric}: {train_result:.4f}/{val_result:.4f}/{test_result:.4f}/{best_val_result:.4f}/{final_test_result:.4f}"
        )


def print_final_results(best_val_result, final_test_result, args):
    print(f"{'*' * 50}\nBest val {args.metric}: {best_val_result}, Final test {args.metric}: {final_test_result}\n{'*' * 50}")


def infer_model(graph, feat, save_path):
    """Load a serialized model and run inference (legacy).

    PyTorch 2.6+ defaults torch.load(weights_only=True), which cannot load a
    pickled nn.Module object. If a legacy `model.pt` exists, load it with
    weights_only=False (trusted checkpoint only).

    For the UMAG usage, classification(save_path=...) now returns logits directly.
    """
    model_path = os.path.join(save_path, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing legacy checkpoint {model_path}. If you only saved weights (state_dict), "
            "run inference with an in-memory model instance instead."
        )
    load_model = th.load(model_path, map_location=feat.device, weights_only=False)
    load_model.eval()
    with th.no_grad():
        pred = load_model(graph, feat)
    print('The prediction files is made.')
    return pred


def mag_train(model, graph, text_feat, visual_feat, labels, train_idx, optimizer, label_smoothing):
    model.train()

    optimizer.zero_grad()
    pred = model(graph, text_feat, visual_feat)
    loss = cross_entropy(pred[train_idx], labels[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def mag_evaluate(
        model, graph, text_feat, visual_feat, labels, train_idx, val_idx, test_idx, metric='accuracy',
        label_smoothing=0.1, average=None, return_pred=False
):
    model.eval()
    with th.no_grad():
        pred = model(graph, text_feat, visual_feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx], label_smoothing)
    test_loss = cross_entropy(pred[test_idx], labels[test_idx], label_smoothing)

    train_results = get_metric(th.argmax(pred[train_idx], dim=1), labels[train_idx], metric, average=average)
    val_results = get_metric(th.argmax(pred[val_idx], dim=1), labels[val_idx], metric, average=average)
    test_results = get_metric(th.argmax(pred[test_idx], dim=1), labels[test_idx], metric, average=average)

    if return_pred:
        return train_results, val_results, test_results, val_loss, test_loss, pred
    return train_results, val_results, test_results, val_loss, test_loss


def mag_classification(
        args, graph, observe_graph, model, text_feat, visual_feat, labels, train_idx, val_idx, test_idx, n_running,
        return_extra=False):
    if args.early_stop_patience is not None:
        stopper = EarlyStopping(patience=args.early_stop_patience)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
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

    # training loop
    total_time = 0
    best_val_result, final_test_result, best_val_loss = 0, 0, float("inf")
    best_val_score = -1.0
    select_metric = getattr(args, "metric", "accuracy")
    select_average = _normalize_average(getattr(args, "average", None))
    report_drop = bool(getattr(args, "report_drop_modality", False))
    report_drop_mode = str(getattr(args, "report_drop_mode", "best")).lower()
    degrade_target = str(getattr(args, "degrade_target", "both")).lower()
    degrade_alphas = _parse_degrade_alphas(args)
    best_test_degrade = None
    best_state_dict = None

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        if args.warmup_epochs is not None:
            adjust_learning_rate(optimizer, args.lr, epoch, args.warmup_epochs)

        train_loss, pred = mag_train(
            model, observe_graph, text_feat, visual_feat, labels, train_idx, optimizer,
            label_smoothing=args.label_smoothing
        )
        if epoch % args.eval_steps == 0:
            (
                train_result,
                val_result,
                test_result,
                val_loss,
                test_loss,
                pred,
            ) = mag_evaluate(
                model,
                graph,
                text_feat,
                visual_feat,
                labels,
                train_idx,
                val_idx,
                test_idx,
                args.metric,
                args.label_smoothing,
                args.average,
                return_pred=True
            )
            if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                wandb.log({
                    'Train_loss': _as_scalar_float(train_loss),
                    'Val_loss': _as_scalar_float(val_loss),
                    'Test_loss': _as_scalar_float(test_loss),
                    'Train_result': _as_scalar_float(train_result),
                    'Val_result': _as_scalar_float(val_result),
                    'Test_result': _as_scalar_float(test_result),
                })
            lr_scheduler.step(_as_scalar_float(train_loss))

            toc = time.time()
            total_time += toc - tic

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
                if (wandb is not None) and (os.environ.get("WANDB_DISABLED", "").lower() not in ("true", "1", "yes")):
                    pass
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_result = val_result
                final_test_result = test_result
                # Keep a CPU copy of weights to avoid GPU memory growth.
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
                    break

            if epoch % args.log_every == 0:
                avg_epoch_time = float(total_time) / float(epoch)
                train_loss_f = _as_scalar_float(train_loss)
                val_loss_f = _as_scalar_float(val_loss)
                test_loss_f = _as_scalar_float(test_loss)
                train_result_f = _as_scalar_float(train_result)
                val_result_f = _as_scalar_float(val_result)
                test_result_f = _as_scalar_float(test_result)
                best_val_result_f = _as_scalar_float(best_val_result)
                final_test_result_f = _as_scalar_float(final_test_result)
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {avg_epoch_time:.2f}\n"
                    f"Loss: {train_loss_f:.4f}\n"
                    f"Train/Val/Test loss: {train_loss_f:.4f}/{val_loss_f:.4f}/{test_loss_f:.4f}\n"
                    f"Train/Val/Test/Best Val/Final Test {args.metric}: {train_result_f:.4f}/{val_result_f:.4f}/{test_result_f:.4f}/{best_val_result_f:.4f}/{final_test_result_f:.4f}"
                )

    print("*" * 50)
    print(f"Best val  {args.metric}: {best_val_result}, Final test  {args.metric}: {final_test_result}")
    print("*" * 50)
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

    if return_extra:
        extra = {
            "best_val_select": _as_scalar_float(best_val_score),
            "best_val_metric": _as_scalar_float(best_val_result),
            "best_test_metric": _as_scalar_float(final_test_result),
            "best_state_dict": best_state_dict,
        }
        if str(getattr(args, "metric", "")).lower() == "accuracy":
            extra.update(
                {
                    "best_val_acc": _as_scalar_float(best_val_result),
                    "best_test_acc": _as_scalar_float(final_test_result),
                }
            )
        if report_drop and best_test_degrade is not None:
            for alpha in degrade_alphas:
                if alpha not in best_test_degrade:
                    continue
                test_degrade_text, test_degrade_vis = best_test_degrade[alpha]
                tag = _alpha_tag(alpha)
                if test_degrade_text is not None:
                    extra[f"best_test_degrade_text_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_text)
                if test_degrade_vis is not None:
                    extra[f"best_test_degrade_visual_{args.metric}_a{tag}"] = _as_scalar_float(test_degrade_vis)
                if str(getattr(args, "metric", "")).lower() == "accuracy":
                    if test_degrade_text is not None:
                        extra[f"best_test_degrade_text_acc_a{tag}"] = _as_scalar_float(test_degrade_text)
                    if test_degrade_vis is not None:
                        extra[f"best_test_degrade_visual_acc_a{tag}"] = _as_scalar_float(test_degrade_vis)
            if len(degrade_alphas) == 1:
                alpha = degrade_alphas[0]
                if alpha in best_test_degrade:
                    test_degrade_text, test_degrade_vis = best_test_degrade[alpha]
                    if test_degrade_text is not None:
                        extra[f"best_test_degrade_text_{args.metric}"] = _as_scalar_float(test_degrade_text)
                    if test_degrade_vis is not None:
                        extra[f"best_test_degrade_visual_{args.metric}"] = _as_scalar_float(test_degrade_vis)
                    if str(getattr(args, "metric", "")).lower() == "accuracy":
                        if test_degrade_text is not None:
                            extra["best_test_degrade_text_acc"] = _as_scalar_float(test_degrade_text)
                        if test_degrade_vis is not None:
                            extra["best_test_degrade_visual_acc"] = _as_scalar_float(test_degrade_vis)
        return best_val_result, final_test_result, extra

    return best_val_result, final_test_result

