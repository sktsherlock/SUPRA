import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

# ==========================================
# 1. 绘图配置
# ==========================================
# 字体设置 (匹配 LaTeX 风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
plt.rcParams['font.size'] = 12

# 线条与刻度
plt.rcParams['axes.linewidth'] = 1.0       # 坐标轴线宽
plt.rcParams['axes.grid'] = True           # 开启网格
plt.rcParams['grid.alpha'] = 0.3           # 网格透明度
plt.rcParams['grid.linestyle'] = '--'      # 网格样式
plt.rcParams['xtick.direction'] = 'in'     # 刻度向内
plt.rcParams['ytick.direction'] = 'in'

# 导出设置
matplotlib.rcParams['pdf.fonttype'] = 42   # 保证导出PDF字体可编辑
matplotlib.rcParams['ps.fonttype'] = 42

# ==========================================
# 2. 数据准备
# ==========================================
methods = ['EF-GNN', 'LF-GNN', 'Tri-GNN']  # 修正名称格式
x = np.arange(len(methods))
width = 0.22  # 柱宽

# 数据
full_acc = [87.22, 87.01, 87.31]
droptext_acc = [87.22, 76.92, 75.76]
dropvisual_acc = [29.87, 27.31, 29.49]
tgnn = 71.845
tmlp = 46.56

# ==========================================
# 3. 绘图主逻辑
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))  # 宽高比调整为 8:5 更紧凑

colors = ['#7294D4', '#A8DADC', '#F19C79'] 
hatches = ['', '///', '...']

# --- 绘制柱状图 ---
# Full Modality
rects1 = ax.bar(x - width, full_acc, width, 
                label='Full Modality', color=colors[0], 
                edgecolor='black', linewidth=0.8, zorder=3)

# Drop Text (Degraded Text)
rects2 = ax.bar(x, droptext_acc, width, 
                label='Degraded Text', color=colors[1], 
                edgecolor='black', hatch=hatches[1], linewidth=0.8, zorder=3)

# Drop Visual (Degraded Visual)
rects3 = ax.bar(x + width, dropvisual_acc, width, 
                label='Degraded Visual', color=colors[2], 
                edgecolor='black', hatch=hatches[2], linewidth=0.8, zorder=3)

# --- 绘制基准线 (Text-only Baseline) ---
ax.axhline(y=tgnn, color='#444444', linestyle='--', 
           linewidth=1.5, zorder=2)
ax.text(2.45, tgnn, f'T-GNN\n({tgnn:.3f}%)', 
        color='#444444', fontsize=12, va='center', ha='left', 
        fontstyle='italic', fontweight='bold')

ax.axhline(y=tmlp, color='#444444', linestyle='--', 
           linewidth=1.5, zorder=2)
ax.text(2.45, tmlp, f'T-MLP\n({tmlp:.3f}%)', 
        color='#444444', fontsize=12, va='center', ha='left', 
        fontstyle='italic', fontweight='bold')

# ==========================================
# 4. 细节美化
# ==========================================

# 设置坐标轴标签
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', labelpad=10)
# 如果方法名很清楚，X轴 label 可以省略，或者写 "Fusion Methods"
# ax.set_xlabel('Multimodal GNN Methods', fontsize=14, fontweight='bold') 

# 设置刻度
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis='y', labelsize=12)

# 设置范围
ax.set_ylim(0, 105) # 留出一点顶部空间给标签
ax.set_xlim(-0.5, 2.5) # 限制X轴范围，不让右侧留白太多

# 去除上方和右侧的边框 (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 图例设置 (放在顶部或右上角)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=3, frameon=False, fontsize=13, columnspacing=1.5)

# --- 数值标签函数 ---
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # 对于很低的值，防止文字和X轴重叠，可以调整位置，或者保持在上方
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移 3 points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# ==========================================
# 5. 保存
# ==========================================
plt.tight_layout()

base_dir = os.path.dirname(os.path.abspath(__file__))
output_pdf = os.path.join(base_dir, 'figures/mmcmp.pdf')
output_png = os.path.join(base_dir, 'figures/mmcmp.png')

plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')

print(f"Chart saved to: {output_png}")
plt.show()