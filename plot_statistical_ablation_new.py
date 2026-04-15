import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Elsevier single-column figure: ~88mm wide = ~3.46 inches
# For full-width (textwidth): ~180mm = ~7.08 inches
# We target full-width with tall aspect to give text room
plt.style.use('seaborn-v0_8-whitegrid')

# Larger base font scale for paper embedding
sns.set_context("paper", font_scale=1.0)

# Force matplotlib font sizes explicitly — overrides seaborn context
plt.rcParams.update({
    'font.size':         14,
    'axes.titlesize':    15,
    'axes.labelsize':    14,
    'xtick.labelsize':   13,
    'ytick.labelsize':   13,
    'legend.fontsize':   12,
    'figure.titlesize':  16,
})

def plot_statistical_ablation():

    # ── Data ─────────────────────────────────────────────────────────────
    conditions = ['Clear\n(ref.)', 'Night', 'Rain', 'Rain+Night']
    means      = [0.8330,           0.7739,  0.8235,  0.7603]
    stds       = [0.0450,           0.0582,  0.0498,  0.0659]

    p_labels = [
        '—',
        r'$p \approx 0$' + '\n(Welch t)',
        r'$p = 1.84 \times 10^{-30}$' + '\n(Welch t)',
        r'$p = 1.12 \times 10^{-112}$' + '\n(Welch t)',
    ]
    cohens_d    = ['—', 'd = 1.195', 'd = 0.202', 'd = 1.547']
    effect_tier = ['—', '[Large]',   '[Small]',    '[Large]']

    shuffle_mean = 0.526
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']

    # ── Figure — wide and tall so text has room ───────────────────────────
    # 14×8 inches at 300 DPI → 4200×2400 px — crisp at any paper scale
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)

    x_pos = np.arange(len(conditions))
    bars = ax.bar(
        x_pos, means, yerr=stds,
        align='center', alpha=0.85,
        ecolor='#333333', capsize=10,
        color=colors, edgecolor='black', linewidth=1.4,
        error_kw=dict(elinewidth=2.0, capthick=2.0)
    )

    # ── Shuffle control line ──────────────────────────────────────────────
    ax.axhline(y=shuffle_mean, color='#222222', linestyle='--',
               linewidth=2.2, zorder=0)
    ax.text(0.99, shuffle_mean,
            r'Spatial shuffle control  $|\rho| = 0.526$',
            ha='right', va='bottom',
            fontsize=13, fontweight='bold', color='#222222',
            transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=3))

    # ── Delta annotation — left margin, outside the green bar ────────────
    ax.annotate('',
                xy=(0, shuffle_mean), xytext=(0, means[0]),
                xycoords=('axes fraction', 'data'),
                textcoords=('axes fraction', 'data'),
                arrowprops=dict(arrowstyle='<->', color='#444444',
                                lw=1.8, shrinkA=0, shrinkB=0))
    ax.text(0.125, (means[0] + shuffle_mean) / 2,
            r'$\Delta|\rho| = 0.307$',
            ha='right', va='center',
            fontsize=12, color='#444444',
            transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))

    # ── Mean labels inside bars ───────────────────────────────────────────
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h / 2,
                r'$|\rho|$' + f' = {means[i]:.3f}',
                ha='center', va='center',
                color='white', fontweight='bold', fontsize=14,
                bbox=dict(facecolor='black', alpha=0.22,
                          edgecolor='none', pad=2))

    # ── Statistical annotations above error bars ──────────────────────────
    for i in range(1, len(bars)):
        bar  = bars[i]
        top  = bar.get_height() + stds[i] + 0.022
        # p-value on first line, Cohen's d + tier on second
        ax.text(bar.get_x() + bar.get_width() / 2., top,
                f'{p_labels[i]}\n{cohens_d[i]}  {effect_tier[i]}',
                ha='center', va='bottom',
                color='black', fontsize=12, fontweight='bold',
                linespacing=1.5)

    # ── Rain robustness note ──────────────────────────────────────────────
    rain_bar = bars[2]
    ax.text(rain_bar.get_x() + rain_bar.get_width() / 2., 0.10,
            'DINOv2 robust\nto rain',
            ha='center', va='bottom',
            fontsize=13, color='white',
            fontstyle='italic', alpha=1.0)

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_ylabel(r'Canonical Correlation $|\rho|$',
                  fontsize=15, fontweight='bold', labelpad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=14, fontweight='bold')
    ax.set_title(
        'Degradation of VFM Geometric Coherence Under Adverse Conditions\n'
        'DINOv2 ViT-L/14 — Dense LiDAR — nuScenes',
        fontsize=15, fontweight='bold', pad=14
    )
    ax.set_ylim(-0.07, 1.08)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.grid(False)

    # ── n labels ─────────────────────────────────────────────────────────
    n_labels = ['n = 7,275', 'n = 3,345', 'n = 6,028', 'n = 642']
    for i, txt in enumerate(n_labels):
        ax.text(i, -0.045, txt,
                ha='center', va='top',
                fontsize=11, color='#555555')

    # ── Save at high DPI — critical for paper embedding ───────────────────
    plt.savefig("statistical_ablation_barchart.png",
                dpi=400, bbox_inches='tight')
    plt.savefig("statistical_ablation_barchart.pdf",
                bbox_inches='tight')
    print("[SUCCESS] Saved .png (400 DPI) and .pdf vector version")

if __name__ == "__main__":
    plot_statistical_ablation()
