import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# ============ 配置 ============
output_dir = r'results/4_LASSO_LOGISTIC'
os.makedirs(output_dir, exist_ok=True)

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10


def save_fig(fig, name):
    fig.savefig(os.path.join(output_dir, f'{name}.tiff'),
                dpi=300, format='tiff', bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(os.path.join(output_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)



data = pd.read_excel('2026-02-24重新清洗前数据_备份.xlsx')

event_col = 'Death'


data[event_col] = data[event_col].replace('null', 0)
data[event_col] = pd.to_numeric(data[event_col], errors='coerce').fillna(0).astype(int)

features_12 = [
    'Mentalillness', 'Age', 'Hypertension', 'Gender',
    'CRPmgL', 'LDLCholesterolmgdL', 'BMI', 'HDLCholesterolmgdL',
    'HF', 'Drink', 'GlycatedHemoglobin', 'CKD'
]

)
binary_vars = ['HF', 'CKD', 'Hypertension', 'Mentalillness', 'Drink']
for var in binary_vars:
    if var in data.columns:
        data[var] = pd.to_numeric(data[var], errors='coerce').map({1: 1, 2: 0, 0: 0})

if 'Gender' in data.columns:
    data['Gender'] = pd.to_numeric(data['Gender'], errors='coerce').map({1: 1, 2: 0})

for f in features_12:
    if f in data.columns:
        data[f] = pd.to_numeric(data[f].replace('null', np.nan), errors='coerce')


df_analysis = data[features_12 + [event_col]].dropna()

X = df_analysis[features_12]
y = df_analysis[event_col]

print(f"分析样本量: {len(df_analysis)}, 死亡事件: {y.sum()} / 存活: {len(y) - y.sum()}")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features_12)

print("\n" + "=" * 60)
print("LASSO Logistic 回归（5折交叉验证）")
print("=" * 60)


alphas = np.logspace(-3, 2, 100)  # 生成 100 个平滑过渡的 lambda (alpha) 值
Cs = 1.0 / alphas  # 转换为 sklearn 需要的 C 值

alphas = np.sort(alphas)
Cs = 1.0 / alphas

coefs = np.zeros((len(Cs), len(features_12)))
# 设置求解器为 saga，适合 L1 惩罚且支持平滑收敛
lr_path = LogisticRegression(penalty='l1', solver='saga', warm_start=True,
                             max_iter=10000, random_state=42)

for i, c in enumerate(Cs):
    lr_path.set_params(C=c)
    lr_path.fit(X_scaled, y)
    coefs[i, :] = lr_path.coef_[0]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_deviance = np.zeros((len(Cs), 5))

for i, c in enumerate(Cs):
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
        X_tr, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lr_cv = LogisticRegression(penalty='l1', solver='saga', C=c,
                                   max_iter=5000, random_state=42)
        lr_cv.fit(X_tr, y_tr)

        # 计算 Binomial Deviance (与 R 语言 glmnet 保持一致)
        y_pred_prob = lr_cv.predict_proba(X_val)[:, 1]
        # deviance = 2 * log_loss
        dev = 2 * log_loss(y_val, y_pred_prob, labels=[0, 1])
        cv_deviance[i, fold_idx] = dev

cv_means = cv_deviance.mean(axis=1)
cv_stds = cv_deviance.std(axis=1)

# 寻找使得 Deviance 最小的最优 Alpha
best_idx = np.argmin(cv_means)
best_alpha = alphas[best_idx]
best_C = Cs[best_idx]

print(f"\n最优 Alpha (Lambda): {best_alpha:.6f}")
print(f"最小 Binomial Deviance: {cv_means[best_idx]:.4f} ± {cv_stds[best_idx]:.4f}")

# 使用最优参数重拟合模型
lr_final = LogisticRegression(penalty='l1', solver='saga', C=best_C,
                              max_iter=10000, random_state=42)
lr_final.fit(X_scaled, y)

# 提取特征和 OR 值
coef_df = pd.DataFrame({
    'Variable': features_12,
    'Coefficient': lr_final.coef_[0],
    'OR': np.exp(lr_final.coef_[0])
})

selected = coef_df[coef_df['Coefficient'].abs() > 1e-6].sort_values('Coefficient', key=abs, ascending=False)
print(f"\n最终保留的特征 ({len(selected)}/12):")
for _, row in selected.iterrows():
    print(f"  {row['Variable']}: Coef={row['Coefficient']:.4f}, OR={row['OR']:.4f}")

fig, ax = plt.subplots(figsize=(8, 5.5))
log_alphas = np.log10(alphas)

# 使用更柔和的颜色映射
cmap = plt.cm.tab20(np.linspace(0, 1, len(features_12)))

for i, feat in enumerate(features_12):
    # 美化标签
    label = feat.replace('mgdL', ' (mg/dL)').replace('mgL', ' (mg/L)')
    ax.plot(log_alphas, coefs[:, i], color=cmap[i], label=label, linewidth=1.5)

ax.axvline(np.log10(best_alpha), color='k', linestyle='--', linewidth=1.2,
           label=f'Best α={best_alpha:.4f}')

ax.set_xlabel('log₁₀(λ)', fontsize=12)
ax.set_ylabel('Coefficients', fontsize=12)
ax.set_title('LASSO Logistic Coefficient Path (12 Features)', fontsize=13)

# 调整图例到外面
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, 'lasso_logistic_path_smooth')

# ============ 图2: 交叉验证偏差图 (Binomial Deviance) ============
fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(log_alphas, cv_means, yerr=cv_stds, fmt='o', color='red',
            ecolor='lightgray', elinewidth=1, capsize=2, markersize=3)

ax.axvline(np.log10(best_alpha), color='k', linestyle='--',
           label=f'Min Deviance at log₁₀(λ)={np.log10(best_alpha):.2f}')

ax.set_xlabel('log₁₀(λ)', fontsize=12)
ax.set_ylabel('Binomial Deviance', fontsize=12)
ax.set_title('LASSO Logistic Cross-Validation', fontsize=13)
ax.legend(fontsize=10, frameon=True)
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig(fig, 'lasso_logistic_cv_curve')

with pd.ExcelWriter(os.path.join(output_dir, 'lasso_logistic_results.xlsx')) as writer:
    coef_df.to_excel(writer, sheet_name='All Features', index=False)
    selected.to_excel(writer, sheet_name='Selected Features', index=False)

print(f"\n✓ 运行完成！平滑的曲线图和结果已保存至: {output_dir} 文件夹。")