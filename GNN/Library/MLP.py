import torch.nn as nn
import argparse
import torch as th
import numpy as np
import torch.nn.functional as F
import sys
import os
import time

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GNN.Utils.LossFunction import cross_entropy, get_metric
from GNN.GraphData import load_data, set_seed
from GNN.Utils.model_config import add_common_args
from GNN.Utils.NodeClassification import initialize_early_stopping, initialize_optimizer_and_scheduler, adjust_learning_rate_if_needed,log_results_to_wandb, log_progress, print_final_results
from GNN.Utils.local_log import build_metric_fields, make_target_from_args, already_logged as local_already_logged, upsert_row


LOG_KEYS = [
    "data_name",
    "graph_path",
    "feature",
    "dropout",
    "lr",
    "wd",
    "min_lr",
    "n_hidden",
    "n_layers",
    "warmup_epochs",
    "n_epochs",
    "eval_steps",
    "early_stop_patience",
    "label_smoothing",
    "metric",
    "average",
    "seed",
    "n_runs",
    "train_ratio",
    "val_ratio",
    "fewshots",
    "selfloop",
    "undirected",
    "inductive",
]


def train(model, feat, labels, train_idx, optimizer, label_smoothing):
    model.train()

    optimizer.zero_grad()
    pred = model(feat)
    loss = cross_entropy(pred[train_idx], labels[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(
        model, feat, labels, train_idx, val_idx, test_idx, metric='accuracy', label_smoothing=0.1, average=None
):
    model.eval()
    with th.no_grad():
        pred = model(feat)
    val_loss = cross_entropy(pred[val_idx], labels[val_idx], label_smoothing)
    test_loss = cross_entropy(pred[test_idx], labels[test_idx], label_smoothing)

    train_results = get_metric(th.argmax(pred[train_idx], dim=1), labels[train_idx], metric, average=average)
    val_results = get_metric(th.argmax(pred[val_idx], dim=1), labels[val_idx], metric, average=average)
    test_results = get_metric(th.argmax(pred[test_idx], dim=1), labels[test_idx], metric, average=average)

    return train_results, val_results, test_results, val_loss, test_loss


def classification(args, model, feat, labels, train_idx, val_idx, test_idx, n_running, save_path=None):
    stopper = initialize_early_stopping(args)
    optimizer, lr_scheduler = initialize_optimizer_and_scheduler(args, model)

    total_time = 0
    best_val_result, final_test_result = -1.0, 0
    best_state_dict = None

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        adjust_learning_rate_if_needed(args, optimizer, epoch)

        train_loss, _ = train_model(model, feat, labels, train_idx, optimizer, args)

        if epoch % args.eval_steps == 0:
            train_result, val_result, test_result, val_loss, test_loss = evaluate_model(
                args, model, feat, labels, train_idx, val_idx, test_idx
            )
            log_results_to_wandb(train_loss, val_loss, test_loss, train_result, val_result, test_result)
            lr_scheduler.step(float(train_loss.detach().item()))

            total_time += time.time() - tic

            best_val_result, final_test_result, best_state_dict = update_best_results(
                val_result,
                test_result,
                best_val_result,
                final_test_result,
                save_path,
                model,
                best_state_dict,
            )

            if should_early_stop(stopper, float(val_result)):
                break

            log_progress(
                args,
                epoch,
                n_running,
                total_time,
                train_loss,
                val_loss,
                test_loss,
                train_result,
                val_result,
                test_result,
                best_val_result,
                final_test_result,
            )

    print_final_results(best_val_result, final_test_result, args)

    if save_path is not None:
        if best_state_dict is None:
            best_state_dict = model.state_dict()
        model.load_state_dict(best_state_dict)
        model.eval()
        with th.no_grad():
            pred = model(feat)
        return best_val_result, final_test_result, pred

    return best_val_result, final_test_result


def train_model(model, feat, labels, train_idx, optimizer, args):
    train_loss, train_result = train(model, feat, labels, train_idx, optimizer, label_smoothing=args.label_smoothing)
    return train_loss, train_result


def evaluate_model(args, model, feat, labels, train_idx, val_idx, test_idx):
    train_result, val_result, test_result, val_loss, test_loss = evaluate(
        model, feat, labels, train_idx, val_idx, test_idx, args.metric, args.label_smoothing, args.average
    )
    return train_result, val_result, test_result, val_loss, test_loss



def update_best_results(val_result, test_result, best_val_result, final_test_result, save_path, model, best_state_dict):
    """Select best checkpoint by validation metric (val_result).

    This keeps MLP consistent with GNN baselines that pick the best epoch by
    validation metric rather than validation loss.
    """
    if val_result > best_val_result:
        # Keep a CPU copy of weights to avoid GPU memory growth.
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            # Save weights-only to avoid pickle / torch.load(weights_only) issues.
            th.save(best_state_dict, os.path.join(save_path, "model_state.pt"))
        return val_result, test_result, best_state_dict
    return best_val_result, final_test_result, best_state_dict


def should_early_stop(stopper, val_result):
    return stopper and stopper.step(val_result)


class MLP(nn.Module):
    def __init__(
            self,
            in_feats,
            n_classes,
            n_layers,
            n_hidden,
            activation,
            dropout=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes

            self.linears.append(nn.Linear(in_hidden, out_hidden))
            if i < n_layers - 1:
                self.norms.append(nn.LayerNorm(out_hidden))

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, feat):
        h = feat

        for i in range(self.n_layers - 1):
            h = self.linears[i](h)
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)

        return self.linears[-1](h)


def args_init():
    argparser = argparse.ArgumentParser(
        "MLP Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(argparser)
    argparser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging for offline runs")
    argparser.add_argument("--local_log", type=str, default=None, help="Optional CSV file to upsert local results")
    return argparser


def main():
    argparser = args_init()
    args = argparser.parse_args()

    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.init(config=args, reinit=True)

    if args.local_log:
        target = make_target_from_args(args, LOG_KEYS)
        if local_already_logged(args.local_log, LOG_KEYS, target):
            print("Experiment with the same hyperparameters already logged in CSV. Skip run.")
            return

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() else 'cpu')

    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio, name=args.data_name,
                                                            fewshots=args.fewshots)

    feat = th.from_numpy(np.load(args.feature).astype(np.float32)).to(device) if args.feature is not None else \
    graph.ndata['feat'].to(device)
    n_classes = (labels.max() + 1).item()
    print(f"Number of classes {n_classes}, Number of features {feat.shape[1]}")

    in_features = feat.shape[1]

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    print(f'Train_idx: {len(train_idx)}')
    print(f'Valid_idx: {len(val_idx)}')
    print(f'Test_idx: {len(test_idx)}')

    labels = labels.to(device)

    # run
    val_results = []
    test_results = []

    # Model implementation

    model = MLP(in_features, n_classes, args.n_layers, args.n_hidden, F.relu, args.dropout).to(device)

    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )
    print(f"Number of the all GNN model params: {TRAIN_NUMBERS}")

    for run in range(args.n_runs):
        set_seed(args.seed + run)
        model.reset_parameters()
        val_result, test_result = classification(
            args, model, feat, labels, train_idx, val_idx, test_idx, run + 1
        )
        if not args.disable_wandb:
            wandb.log({f'Val_{args.metric}': val_result, f'Test_{args.metric}': test_result})
        val_results.append(val_result)
        test_results.append(test_result)

    print(f"Runned {args.n_runs} times")
    print(f"Average val {args.metric}: {np.mean(val_results)} ± {np.std(val_results)}")
    print(f"Average test {args.metric}: {np.mean(test_results)} ± {np.std(test_results)}")
    if not args.disable_wandb:
        wandb.log({f'Mean_Val_{args.metric}': np.mean(val_results), f'Mean_Test_{args.metric}': np.mean(test_results)})

    if args.local_log:
        row = {k: str(getattr(args, k, "")) for k in LOG_KEYS}
        row.update(build_metric_fields("val", val_results))
        row.update(build_metric_fields("test", test_results))
        upsert_row(args.local_log, LOG_KEYS, row)


if __name__ == "__main__":
    main()
