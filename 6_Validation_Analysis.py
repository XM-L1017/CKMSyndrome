import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore')


def load_and_clean_data(file_path):
    print(f"正在读取数据: '{file_path}' ...")
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 找不到文件 '{file_path}'")

    data.columns = data.columns.str.strip()

    target_col = None
    for col in data.columns:
        if col.lower() in ['death', 'isdeath']:
            target_col = col
            break

    if not target_col:
        raise ValueError(f"在 {file_path} 中找不到 Death 或 Isdeath 列")

    data = data.rename(columns={target_col: 'Target'})
    data['Target'] = data['Target'].replace('null', 0)
    data['Target'] = pd.to_numeric(data['Target'], errors='coerce').fillna(0).astype(int)

    binary_vars = ['HF', 'CKD', 'Hypertension', 'Mentalillness', 'Drink']
    for var in binary_vars:
        if var in data.columns:
            data[var] = pd.to_numeric(data[var], errors='coerce').map({1: 1, 2: 0, 0: 0})

    if 'Gender' in data.columns:
        data['Gender'] = pd.to_numeric(data['Gender'], errors='coerce').map({1: 1, 2: 0})

    return data



train_file = '2026-02-24重新清洗前数据_备份.xlsx'
test_file = 'External validation data.xlsx'

print("【步骤 1：处理数据集】")
df_train = load_and_clean_data(train_file)
df_test = load_and_clean_data(test_file)

features_12 = [
    'Mentalillness', 'Age', 'Hypertension', 'Gender',
    'CRPmgL', 'LDLCholesterolmgdL', 'BMI', 'HDLCholesterolmgdL',
    'HF', 'Drink', 'GlycatedHemoglobin', 'CKD'
]

common_features = [f for f in features_12 if f in df_train.columns and f in df_test.columns]

df_train = df_train[common_features + ['Target']].dropna()
df_test = df_test[common_features + ['Target']].dropna()

X_train, y_train = df_train[common_features], df_train['Target']
X_test, y_test = df_test[common_features], df_test['Target']

print(f"✓ 训练集有效样本量: {len(X_train)}, 死亡事件: {y_train.sum()}")
print(f"✓ 验证集有效样本量: {len(X_test)}, 死亡事件: {y_test.sum()}\n")


base_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=2,
    random_state=42,
    class_weight='balanced'
)


calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method='sigmoid',
    cv=5
)
calibrated_model.fit(X_train, y_train)

y_prob = calibrated_model.predict_proba(X_test)[:, 1]


mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 6))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 图例中仅保留 AUC 值
ax.plot(fpr, tpr, color='#5D9BCA', lw=2.5,
         label=f'Test Set ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#D9534F', lw=1.5, linestyle='--')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('1 - Specificity', fontsize=12)
ax.set_ylabel('Sensitivity', fontsize=12)
ax.set_title('ROC Curve (Test Set)', fontsize=14)
ax.legend(loc="lower right", frameon=False, fontsize=12)
ax.grid(True, linestyle=':', alpha=0.6)

output_dir = r'results/6_Validation_Analysis'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'LightGBM_External_Validation_ROC.png')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"运行成功！外部验证 ROC 曲线已保存至: {output_path}")
plt.show()