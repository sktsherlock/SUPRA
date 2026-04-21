import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from GNN.GraphData import load_data
from GNN.Utils.graph_stats import compute_graph_statistics, format_graph_statistics


argparser = argparse.ArgumentParser(
    "Homo Config",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
argparser.add_argument(
    "--graph_path", type=str, default=None, help="The datasets to be implemented."
)
argparser.add_argument(
    "--train_ratio", type=float, default=0.6, help="training ratio"
)
argparser.add_argument(
    "--val_ratio", type=float, default=0.2, help="training ratio"
)
argparser.add_argument(
    "--data_name", type=str, default=None, help="The dataset name.",
)
args = argparser.parse_args()

graph, labels, train_idx, val_idx, test_idx = load_data(
    args.graph_path,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    name=args.data_name,
)

stats = compute_graph_statistics(graph, labels)
print(format_graph_statistics(stats))
