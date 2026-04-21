import sys
import copy
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import argparse
import wandb
import torch as th
import numpy as np
import torch.nn.functional as F
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GNN.GraphData import load_data, set_seed
from GNN.Utils.NodeClassification import classification
from GNN.Utils.model_config import add_common_args
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


# 模型定义模块
class GCN(nn.Module):
    def __init__(
            self,
            in_feats,
            n_hidden,
            n_classes,
            n_layers,
            activation,
            dropout,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            self.convs.append(
                dglnn.GraphConv(in_hidden, out_hidden, "both")
            )

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, graph, feat):
        h = feat

        for i in range(self.n_layers):
            h = self.convs[i](graph, h)

            if i < self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)

        return h


# 参数定义模块
def args_init():
    argparser = argparse.ArgumentParser(
        "GCN Config",
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
        wandb.init(config=args, reinit=True, entity="tiant-wang-org")

    if args.local_log:
        target = make_target_from_args(args, LOG_KEYS)
        if local_already_logged(args.local_log, LOG_KEYS, target):
            print("Experiment with the same hyperparameters already logged in CSV. Skip run.")
            return

    device = th.device("cuda:%d" % args.gpu if th.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load data
    graph, labels, train_idx, val_idx, test_idx = load_data(args.graph_path, train_ratio=args.train_ratio,
                                                            val_ratio=args.val_ratio, name=args.data_name,
                                                            fewshots=args.fewshots)

    # add reverse edges, tranfer to the  undirected graph
    if args.undirected:
        print("The Graph change to the undirected graph")
        srcs, dsts = graph.all_edges()
        graph.add_edges(dsts, srcs)

    # 定义可观测图数据，用于inductive实验设置；
    observe_graph = copy.deepcopy(graph)

    if args.inductive:
        # 构造Inductive Learning 实验条件

        isolated_nodes = th.cat((val_idx, test_idx))
        sort_isolated_nodes, _ = th.sort(isolated_nodes)
        # 从图中删除指定节点
        observe_graph.remove_nodes(sort_isolated_nodes)

        # 添加相同数量的孤立节点
        observe_graph.add_nodes(len(sort_isolated_nodes))
        print(observe_graph)
        print('***************')
        print(graph)

    # add self-loop
    if args.selfloop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
        observe_graph = observe_graph.remove_self_loop().add_self_loop()

    feat = th.from_numpy(np.load(args.feature).astype(np.float32)).to(device) if args.feature is not None else \
        graph.ndata['feat'].to(device)
    n_classes = (labels.max() + 1).item()
    print(f"Number of classes {n_classes}, Number of features {feat.shape[1]}")

    graph.create_formats_()
    observe_graph.create_formats_()

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    print(f'Train_idx: {len(train_idx)}')
    print(f'Valid_idx: {len(val_idx)}')
    print(f'Test_idx: {len(test_idx)}')

    labels = labels.to(device)
    graph = graph.to(device)
    observe_graph = observe_graph.to(device)


    # Model implementation
    model = GCN(feat.shape[1], args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout).to(device)
    TRAIN_NUMBERS = sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )
    print(f"Number of the all GNN model params: {TRAIN_NUMBERS}")

    # run
    val_results = []
    test_results = []
    
    for run in range(args.n_runs):
        set_seed(args.seed + run)
        model.reset_parameters()
        val_result, test_result = classification(
            args, graph, observe_graph, model, feat, labels, train_idx, val_idx, test_idx, run + 1
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
