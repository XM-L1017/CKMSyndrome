
import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

output_dir = r'results/2_COX'
os.makedirs(output_dir, exist_ok=True)

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10

data = pd.read_excel('2026-02-24重新清洗前数据_备份.xlsx')


duration_col = 'survival time （year）'
event_col = 'Death'


data[event_col] = data[event_col].replace('null', 0)  # null → 0（存活）
data[event_col] = pd.to_numeric(data[event_col], errors='coerce').fillna(0).astype(int)

other_cols = [c for c in data.columns if c != event_col]
data[other_cols] = data[other_cols].replace('null', np.nan)


data[duration_col] = pd.to_numeric(data[duration_col], errors='coerce')
max_followup = data[duration_col].max()
print(f"最大随访时间: {max_followup} 年")


mask_missing_time = data[duration_col].isna()
mask_alive = data[event_col] == 0
data.loc[mask_missing_time & mask_alive, duration_col] = max_followup

data = data[~(mask_missing_time & (data[event_col] == 1))]

data = data[data[duration_col] > 0]

print(f"\n清洗后样本量: {len(data)}")
print(f"死亡事件: {data[event_col].sum()} ({data[event_col].mean()*100:.1f}%)")
print(f"存活/截尾: {(data[event_col]==0).sum()} ({(1-data[event_col].mean())*100:.1f}%)")
print(f"中位随访时间: {data[duration_col].median():.1f} 年")
print(f"随访时间范围: {data[duration_col].min():.1f} - {data[duration_col].max():.1f} 年")


binary_recode_vars = [
    'DM', 'HF', 'CKD', 'Hypertension', 'dyslipidemia', 'Mentalillness',
    'Smoke', 'Drink'
]
for var in binary_recode_vars:
    if var in data.columns:
        data[var] = pd.to_numeric(data[var], errors='coerce')
        # 1→1, 2→0
        data[var] = data[var].map({1: 1, 2: 0})
        print(f"  {var}: 1(有)={data[var].sum():.0f}, 0(无)={(data[var]==0).sum():.0f}")

# Gender: 1=男, 2=女 → 1=男, 0=女
if 'Gender' in data.columns:
    data['Gender'] = pd.to_numeric(data['Gender'], errors='coerce')
    data['Gender'] = data['Gender'].map({1: 1, 2: 0})
    print(f"  Gender: 男={data['Gender'].sum():.0f}, 女={(data['Gender']==0).sum():.0f}")

# CKM: 保持 1/2/3 作为有序变量（或作为连续变量使用）
if 'CKM' in data.columns:
    data['CKM'] = pd.to_numeric(data['CKM'], errors='coerce')
    print(f"  CKM分布: {data['CKM'].value_counts().sort_index().to_dict()}")


continuous_vars = [
    'Age', 'BMI', 'Waist',
    'Glucose', 'TotalCholesterolmgdL', 'HDLCholesterolmgdL',
    'LDLCholesterolmgdL', 'TriglyceridesmgdL',
    'BUNmgdL', 'CreatininemgdL', 'UricAcidmgdL',
    'CRPmgL', 'GlycatedHemoglobin',
]
for var in continuous_vars:
    if var in data.columns:
        data[var] = pd.to_numeric(data[var], errors='coerce')


categorical_vars = [
    'CKM', 'Gender', 'DM', 'HF', 'CKD',
    'Hypertension', 'dyslipidemia', 'Mentalillness',
    'Smoke', 'Drink', 'Marriage', 'Education', 'Living area',
]

all_vars = categorical_vars + continuous_vars
all_vars = [v for v in all_vars if v in data.columns]


print("\n" + "=" * 60)
print("1. 单因素 Cox 比例风险回归分析")
print("=" * 60)

results = []
significant_vars = []

for var in all_vars:
    try:
        df_temp = data[[var, duration_col, event_col]].dropna()

        if df_temp[var].nunique() < 2:
            print(f"  ⚠ 变量 {var} 只有1个唯一值，跳过")
            continue

        cph = CoxPHFitter()
        cph.fit(df_temp, duration_col=duration_col, event_col=event_col)

        summary = cph.summary
        hr = summary.loc[var, 'exp(coef)']
        ci_lower = summary.loc[var, 'exp(coef) lower 95%']
        ci_upper = summary.loc[var, 'exp(coef) upper 95%']
        p_value = summary.loc[var, 'p']

        if p_value <= 0.05:
            significant_vars.append(var)
            flag = ' ***'
        elif p_value <= 0.1:
            flag = ' *'
        else:
            flag = ''

        results.append({
            'Variable': var,
            'HR': round(hr, 4),
            'HR_Lower_95CI': round(ci_lower, 4),
            'HR_Upper_95CI': round(ci_upper, 4),
            'HR_95CI': f"{ci_lower:.3f}-{ci_upper:.3f}",
            'P_value': round(p_value, 4),
            'Significant': '是' if p_value <= 0.05 else '否',
            'N': len(df_temp),
            'Events': int(df_temp[event_col].sum())
        })
        print(f"  {var}: HR={hr:.3f} ({ci_lower:.3f}-{ci_upper:.3f}), P={p_value:.4f}{flag}")

    except Exception as e:
        print(f"  ✗ 变量 {var} 分析失败: {e}")
        results.append({
            'Variable': var, 'HR': np.nan, 'HR_Lower_95CI': np.nan,
            'HR_Upper_95CI': np.nan, 'HR_95CI': 'N/A',
            'P_value': np.nan, 'Significant': 'N/A', 'N': 0, 'Events': 0
        })

results_df = pd.DataFrame(results)
print(f"\n显著变量 (P≤0.05): {', '.join(significant_vars) if significant_vars else '无'}")
results_df.to_excel(os.path.join(output_dir, 'univariate_cox_results.xlsx'), index=False)

# ============ 2. 多因素 Cox 回归 ============
print("\n" + "=" * 60)
print("2. 多因素 Cox 比例风险回归分析")
print("=" * 60)

multi_results_df = None
if len(significant_vars) >= 2:
    df_multi = data[significant_vars + [duration_col, event_col]].dropna()
    print(f"  纳入变量: {significant_vars}")
    print(f"  有效样本: {len(df_multi)}, 事件: {df_multi[event_col].sum():.0f}")

    cph_multi = CoxPHFitter()
    cph_multi.fit(df_multi, duration_col=duration_col, event_col=event_col)

    multi_summary = cph_multi.summary.reset_index()
    multi_results_df = pd.DataFrame({
        'Variable': multi_summary['covariate'],
        'HR': multi_summary['exp(coef)'].round(4),
        'HR_Lower_95CI': multi_summary['exp(coef) lower 95%'].round(4),
        'HR_Upper_95CI': multi_summary['exp(coef) upper 95%'].round(4),
        'P_value': multi_summary['p'].round(4),
        'Significant': multi_summary['p'].apply(lambda x: '是' if x <= 0.05 else '否')
    })

    print("\n多因素Cox回归结果:")
    for _, row in multi_results_df.iterrows():
        print(f"  {row['Variable']}: HR={row['HR']:.3f} "
              f"({row['HR_Lower_95CI']:.3f}-{row['HR_Upper_95CI']:.3f}), "
              f"P={row['P_value']:.4f} {'***' if row['P_value']<=0.05 else ''}")

    multi_results_df.to_excel(os.path.join(output_dir, 'multivariate_cox_results.xlsx'), index=False)

    # 比例风险假设检验
    print("\n  比例风险假设检验:")
    try:
        cph_multi.check_assumptions(df_multi, p_value_threshold=0.05, show_plots=False)
        print("  → 比例风险假设满足")
    except Exception as e:
        print(f"  → {e}")

elif len(significant_vars) == 1:
    print(f"  仅1个显著变量 ({significant_vars[0]})，可跳过或报告单因素结果")
else:
    print("  无显著变量，跳过多因素分析")

# ============ 3. 森林图 ============
print("\n" + "=" * 60)
print("3. 绘制森林图")
print("=" * 60)

# 对所有变量绘制森林图（不仅仅是显著的）
plot_df = results_df.dropna(subset=['HR']).copy()
plot_df = plot_df.sort_values('P_value')  # 按 P 值排序

if len(plot_df) > 0:
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.4)))

    y_pos = range(len(plot_df))
    hrs = plot_df['HR'].values
    ci_low = plot_df['HR_Lower_95CI'].values
    ci_high = plot_df['HR_Upper_95CI'].values
    labels = plot_df['Variable'].values
    p_vals = plot_df['P_value'].values

    for i in range(len(plot_df)):
        is_sig = p_vals[i] <= 0.05
        color = '#e74c3c' if is_sig else '#95a5a6'  # 显著=红色，不显著=灰色
        linewidth = 2.5 if is_sig else 1.5
        markersize = 8 if is_sig else 5

        ax.plot([ci_low[i], ci_high[i]], [i, i], color=color, linewidth=linewidth)
        ax.plot(hrs[i], i, 'o', color=color, markersize=markersize, zorder=5)

    ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    ax.set_title('Univariate Cox Regression - Forest Plot\n(Red = P≤0.05, Gray = P>0.05)', fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2)

    # 右侧标注
    x_max = ax.get_xlim()[1]
    for i in range(len(plot_df)):
        p_str = f'P={p_vals[i]:.3f}' if p_vals[i] >= 0.001 else 'P<0.001'
        ax.text(x_max * 1.02, i,
                f'HR={hrs[i]:.2f} ({ci_low[i]:.2f}-{ci_high[i]:.2f}) {p_str}',
                va='center', fontsize=7,
                color='#e74c3c' if p_vals[i] <= 0.05 else '#666666')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forest_plot_univariate.tiff'),
                dpi=300, format='tiff', bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.savefig(os.path.join(output_dir, 'forest_plot_univariate.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 森林图已保存")

# ============ 4. Kaplan-Meier 生存曲线 ============
print("\n" + "=" * 60)
print("4. Kaplan-Meier 生存曲线")
print("=" * 60)

km_configs = [
    {
        'var': 'CKM',
        'title': 'CKM Stage',
        'labels': {1: 'CKM Stage 1', 2: 'CKM Stage 2', 3: 'CKM Stage 3'},
    },
    {
        'var': 'Hypertension',
        'title': 'Hypertension',
        'labels': {1: 'Hypertension', 0: 'No Hypertension'},  # 已重编码
    },
    {
        'var': 'HF',
        'title': 'Heart Failure (CVD)',
        'labels': {1: 'HF', 0: 'No HF'},  # 已重编码
    },
    {
        'var': 'Mentalillness',
        'title': 'Mental Illness',
        'labels': {1: 'Mental Illness', 0: 'No Mental Illness'},  # 已重编码
    },
]

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

for config in km_configs:
    strat_var = config['var']
    if strat_var not in data.columns:
        print(f"  ⚠ {strat_var} 不在数据中，跳过")
        continue

    df_km = data[[strat_var, duration_col, event_col]].dropna()
    groups = sorted(df_km[strat_var].unique())

    if len(groups) < 2:
        print(f"  ⚠ {strat_var} 只有{len(groups)}组，跳过")
        continue

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()

    for i, group in enumerate(groups):
        mask = df_km[strat_var] == group
        df_group = df_km.loc[mask]
        label_text = config['labels'].get(group, f'{strat_var}={group}')
        n_events = int(df_group[event_col].sum())
        label_text += f' (n={len(df_group)}, events={n_events})'

        kmf.fit(df_group[duration_col], event_observed=df_group[event_col],
                label=label_text)
        kmf.plot_survival_function(ax=ax, color=colors[i % len(colors)],
                                    linewidth=2, ci_show=True, ci_alpha=0.15)

    # Log-rank 检验
    if len(groups) == 2:
        g1 = df_km[df_km[strat_var] == groups[0]]
        g2 = df_km[df_km[strat_var] == groups[1]]
        lr = logrank_test(g1[duration_col], g2[duration_col],
                          event_observed_A=g1[event_col], event_observed_B=g2[event_col])
        p_val = lr.p_value
    else:
        lr = multivariate_logrank_test(df_km[duration_col], df_km[strat_var], df_km[event_col])
        p_val = lr.p_value

    p_text = f'Log-rank P = {p_val:.4f}' if p_val >= 0.001 else 'Log-rank P < 0.001'

    ax.text(0.95, 0.05, p_text, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Kaplan-Meier Survival Curve by {config["title"]}', fontsize=13)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax.grid(alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    plt.tight_layout()

    fname = f'KM_curve_{strat_var}'
    plt.savefig(os.path.join(output_dir, f'{fname}.tiff'),
                dpi=300, format='tiff', bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.savefig(os.path.join(output_dir, f'{fname}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname} ({p_text})")

# ============ 5. 保存汇总 ============
with pd.ExcelWriter(os.path.join(output_dir, 'cox_analysis_all_results.xlsx')) as writer:
    results_df.to_excel(writer, sheet_name='Univariate Cox', index=False)
    if multi_results_df is not None:
        multi_results_df.to_excel(writer, sheet_name='Multivariate Cox', index=False)
    pd.DataFrame({'Significant Variables (P≤0.05)': significant_vars or ['无']}).to_excel(
        writer, sheet_name='Significant Variables', index=False)

print(f"\n✓ 所有结果已保存到: {output_dir}")