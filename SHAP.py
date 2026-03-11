import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

output_dir = r'results/5_SHAP_Analysis'
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



file_name = '2026-02-24重新清洗前数据_备份.xlsx'
try:
    data = pd.read_excel(file_name)
except FileNotFoundError:
    print(f"错误: 找不到文件 '{file_name}'。请确保该 Excel 文件与本 Python 脚本在同一个文件夹下。")
    exit()

event_col = 'Death'

data[event_col] = data[event_col].replace('null', 0)
data[event_col] = pd.to_numeric(data[event_col], errors='coerce').fillna(0).astype(int)

features_12 = [
    'Mentalillness', 'Age', 'Hypertension', 'Gender',
    'CRPmgL', 'LDLCholesterolmgdL', 'BMI', 'HDLCholesterolmgdL',
    'HF', 'Drink', 'GlycatedHemoglobin', 'CKD'
]

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

print(f"✓ 数据清洗完毕！有效样本量: {len(X)}, 死亡事件: {y.sum()}")


best_model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight='balanced')
best_model.fit(X, y)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)

if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

beautiful_features = [f.replace('mgdL', ' (mg/dL)').replace('mgL', ' (mg/L)') for f in features_12]
X_plot = X.copy()
X_plot.columns = beautiful_features


plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_to_plot, X_plot, show=False)
plt.title('SHAP Summary Plot (Feature Impact on Death Risk)', fontsize=14, pad=15)
fig1 = plt.gcf()
save_fig(fig1, 'shap_summary_dot_plot_REAL')


plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_to_plot, X_plot, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Mean Absolute SHAP Value)', fontsize=14, pad=15)
fig2 = plt.gcf()
save_fig(fig2, 'shap_feature_importance_bar_REAL')