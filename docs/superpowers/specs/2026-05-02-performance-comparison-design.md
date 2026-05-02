# Performance Comparison Plot — Design Spec

## Overview

Create a faceted grouped bar chart script (`GNN/Utils/plot_performance_comparison.py`) for the SUPRA paper Intro, demonstrating that multimodal graph learning methods gain less from high-quality features.

## Data

### Raw Numbers (hardcoded inline, extensible via CSV later)

| Dataset | Encoder   | MLP  | MGCN | MGAT | MIGGT | NTSFormer |
|---------|-----------|------|------|------|-------|-----------|
| Movies  | clip-roberta | 51.29 | 53.95 | 53.88 | 49.89 | 54.07 |
| Movies  | llama     | 57.51 | 55.19 | 54.44 | 55.93 | 56.14 |
| Grocery | clip-roberta | 80.62 | 81.53 | 81.40 | 80.04 | 83.47 |
| Grocery | llama     | 86.51 | 84.56 | 84.28 | 85.82 | 87.03 |
| RedditM | clip-roberta | 83.20 | 78.46 | 78.11 | 82.91 | 86.13 |
| RedditM | llama     | 84.80 | 79.25 | 77.90 | 84.69 | 87.28 |

## Visual Design

### Figure Layout
- 3 subplots in 1 row: Movies | Grocery | RedditM
- Subplot figsize: `figsize=(14, 4.5)`, `gridspec_kw={'wspace': 0.35}`
- Each subplot: grouped bar chart, 5 model groups, 2 bars each

### Color Scheme
| Encoder     | Color     | Hex       |
|-------------|-----------|-----------|
| clip-roberta | blue     | `#4C72B0` |
| llama       | orange   | `#DD8452` |

### Bar Style
- `bar_width = 0.35`
- Gap between clip-roberta / llama within same model group: `0.05`
- `edgecolor='white'`, `linewidth=0.5`
- NTSFormer bars: `edgecolor='#333333'`, `linewidth=1.5` (bold outline to highlight SOTA)

### Y-Axis (independent per subplot)
| Dataset  | Y range |
|----------|---------|
| Movies   | [45, 62] |
| Grocery  | [78, 90] |
| RedditM  | [75, 90] |
- Format: percentage (`0%`, `20%`, `40%`, `60%`, `80%`)
- `yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))`

### X-Axis
- Tick labels: `MLP`, `MGCN`, `MGAT`, `MIGGT`, `NTSFormer`
- Rotation: 0° (horizontal)
- `fontsize=10`

### Labels & Title
- Each subplot title: `Movies`, `Grocery`, `RedditM` (`fontsize=12`, `fontweight='bold'`)
- Y-axis label: `Accuracy (%)` on leftmost subplot only
- Main figure title (optional): not needed — subplot titles are sufficient

### Legend
- Position: below figure, `bbox_to_anchor=(0.5, -0.18)`, `loc='upper center'`
- Labels: `clip-roberta`, `llama`
- `ncol=2`, `framealpha=0.9`

### Grid & Spines
- `ax.yaxis.grid(True, linestyle='--', alpha=0.4)`, `set_axisbelow(True)`
- `spines['top'].set_visible(False)`, `spines['right'].set_visible(False)`

## Output

- PDF vector format: `{save_path}.pdf` (auto-converted from `--save_plot` path)
- `dpi=300`, `bbox_inches='tight'`
- If `--save_plot` not provided, call `plt.show()`

## CLI Interface

```bash
python -m GNN.Utils.plot_performance_comparison \
    --save_plot Results/perf_comparison.pdf
```

## File Location

`GNN/Utils/plot_performance_comparison.py`

## Dependencies

- `matplotlib`, `numpy` (already in environment)
- Standard library: `argparse`, `os`
