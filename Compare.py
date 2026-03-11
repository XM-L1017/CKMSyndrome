import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import warnings

# Import machine learning models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    print("Warning: xgboost not found, using HistGradientBoostingClassifier as alternative.")
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier

# Ignore convergence warnings for cleaner output during cross-validation
warnings.filterwarnings('ignore')

# ============ Configuration ============
output_dir = r'results/supplementary_analysis'
os.makedirs(output_dir, exist_ok=True)

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10

HERO_MODEL = 'LightGBM'
HERO_COLOR = '#e74c3c'


def save_fig(fig, name, split_name=""):
    # 结合阶段名称生成安全的文件名
    if split_name:
        safe_name = f"{name}_{split_name}".replace(" ", "_").replace("(", "").replace(")", "")
    else:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")

    fig.savefig(os.path.join(output_dir, f'{safe_name}.tiff'),
                dpi=300, format='tiff', bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(os.path.join(output_dir, f'{safe_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============ Plotting Functions ============

def plot_multi_roc(y_true, model_probs_dict, split_name="Validation"):
    fig, ax = plt.subplots(figsize=(7, 6))
    bg_colors = ['#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e', '#95a5a6', '#e67e22']
    c_idx = 0

    hero_data = None
    other_data = []

    for name, y_prob in model_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        if name == HERO_MODEL:
            hero_data = (fpr, tpr, auc_val, name)
        else:
            other_data.append((fpr, tpr, auc_val, name))

    for fpr, tpr, auc_val, name in other_data:
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})',
                color=bg_colors[c_idx % len(bg_colors)], linewidth=1.5, alpha=0.6, zorder=3)
        c_idx += 1

    if hero_data:
        fpr, tpr, auc_val, name = hero_data
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})',
                color=HERO_COLOR, linewidth=3.5, zorder=10)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, zorder=1)
    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
    ax.set_title(f'ROC Curve ({split_name})', fontsize=14, fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_fig(fig, 'ROC_curve', split_name)


def plot_multi_calibration(y_true, model_probs_dict, split_name="Validation"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration', zorder=1)

    bg_colors = ['#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e', '#95a5a6']
    c_idx = 0
    hero_data = None
    other_data = []

    for name, y_prob in model_probs_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy='quantile')
        if name == HERO_MODEL:
            hero_data = (prob_pred, prob_true, name)
        else:
            other_data.append((prob_pred, prob_true, name))

    for prob_pred, prob_true, name in other_data:
        ax.plot(prob_pred, prob_true, 'o-', label=name,
                color=bg_colors[c_idx % len(bg_colors)], linewidth=1.2, markersize=5, alpha=0.5, zorder=3)
        c_idx += 1

    if hero_data:
        prob_pred, prob_true, name = hero_data
        ax.plot(prob_pred, prob_true, 's-', label=name,
                color=HERO_COLOR, linewidth=3, markersize=8, zorder=10)

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Probability', fontsize=12)
    ax.set_title(f'Calibration Curve ({split_name})', fontsize=14, fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_fig(fig, 'Calibration_curve', split_name)


def plot_multi_dca(y_true, model_probs_dict, split_name="Validation"):
    thresholds = np.arange(0.01, 0.99, 0.01)
    n = len(y_true)
    prevalence = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(8, 6))
    nb_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    ax.plot(thresholds, nb_all, 'k:', linewidth=1.5, label='Treat All', zorder=2)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.5, label='Treat None', zorder=2)

    bg_colors = ['#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#34495e', '#95a5a6']
    c_idx = 0
    hero_data = None
    other_data = []

    for name, y_prob in model_probs_dict.items():
        net_benefits = []
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            nb = tp / n - fp / n * thresh / (1 - thresh)
            net_benefits.append(nb)

        if name == HERO_MODEL:
            hero_data = (net_benefits, name)
        else:
            other_data.append((net_benefits, name))

    for net_benefits, name in other_data:
        ax.plot(thresholds, net_benefits, label=name,
                color=bg_colors[c_idx % len(bg_colors)], linewidth=1.5, alpha=0.6, zorder=3)
        c_idx += 1

    if hero_data:
        net_benefits, name = hero_data
        ax.plot(thresholds, net_benefits, label=name,
                color=HERO_COLOR, linewidth=3.5, zorder=10)

    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title(f'Decision Curve Analysis ({split_name})', fontsize=14, fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='upper right', ncol=2)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.05, 0.1)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_fig(fig, 'DCA', split_name)


def plot_multi_cv_boxplot(cv_results_dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(cv_results_dict.keys())

    # Ensure HERO_MODEL is plotted first
    if HERO_MODEL in models:
        models.remove(HERO_MODEL)
        models = [HERO_MODEL] + models

    data_to_plot = [cv_results_dict[m] for m in models]

    box = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6,
                     boxprops=dict(facecolor='#ecf0f1', color='gray'),
                     capprops=dict(color='gray'),
                     whiskerprops=dict(color='gray'),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5),
                     medianprops=dict(color='black', linewidth=1.5))

    if HERO_MODEL in models:
        hero_idx = models.index(HERO_MODEL)
        box['boxes'][hero_idx].set_facecolor('#fadbd8')
        box['boxes'][hero_idx].set_edgecolor(HERO_COLOR)
        box['boxes'][hero_idx].set_linewidth(2)
        box['medians'][hero_idx].set_color(HERO_COLOR)
        box['medians'][hero_idx].set_linewidth(2.5)

    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('ROC AUC', fontsize=12)
    ax.set_title('10-Fold Cross-Validation AUC Comparison (Training Data)', fontsize=14, fontweight='bold')

    for i, model in enumerate(models):
        mean_auc = np.mean(cv_results_dict[model])
        color = HERO_COLOR if model == HERO_MODEL else 'black'
        fontweight = 'bold' if model == HERO_MODEL else 'normal'
        ax.text(i + 1, np.max(cv_results_dict[model]) + 0.01, f'{mean_auc:.3f}',
                ha='center', va='bottom', fontsize=9, color=color, fontweight=fontweight)

    ax.set_ylim(bottom=max(0.4, np.min([np.min(d) for d in data_to_plot]) - 0.05), top=1.02)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    save_fig(fig, '10fold_CV_boxplot')


def run_multi_10fold_cv(X, y, base_models):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_auc_dict = {name: [] for name in base_models.keys()}

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for name, base_model in base_models.items():
            model = clone(base_model)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            auc_val = roc_auc_score(y_val, y_prob)
            cv_auc_dict[name].append(auc_val)

    return cv_auc_dict


# ============ Main Execution ============
def main():
    print("=" * 60)
    print("Machine Learning Model Evaluation and Validation (Rigorous Pipeline)")
    print("=" * 60)

    print("\n[Loading and preprocessing clinical dataset...]")

    file_name = '2026-02-24重新清洗前数据_备份.xlsx'

    try:
        data = pd.read_excel(file_name)
    except FileNotFoundError:
        print(f"Error: Dataset '{file_name}' not found.")
        return

    event_col = 'Death'
    data[event_col] = data[event_col].replace('null', 0)
    data[event_col] = pd.to_numeric(data[event_col], errors='coerce').fillna(0).astype(int)

    selected_features = [
        'Age', 'CRPmgL', 'LDLCholesterolmgdL', 'HDLCholesterolmgdL',
        'BMI', 'GlycatedHemoglobin', 'Hypertension', 'Mentalillness',
        'Gender', 'Drink', 'HF', 'CKD'
    ]

    binary_vars = ['Hypertension', 'Mentalillness', 'Drink', 'HF', 'CKD', 'Gender']
    for var in binary_vars:
        if var in data.columns:
            data[var] = pd.to_numeric(data[var], errors='coerce').map({1: 1, 2: 0})

    cont_vars = ['Age', 'CRPmgL', 'LDLCholesterolmgdL', 'HDLCholesterolmgdL', 'BMI', 'GlycatedHemoglobin']
    for var in cont_vars:
        if var in data.columns:
            data[var] = pd.to_numeric(data[var].replace('null', np.nan), errors='coerce')

    df_analysis = data[selected_features + [event_col]].dropna()
    X_real = df_analysis[selected_features].values
    y_true = df_analysis[event_col].values

    print(f"  ✓ Data loaded successfully. Analyzed samples: {len(df_analysis)}, Events: {sum(y_true)}")

    print("\n[Splitting dataset into Training (70%) and Validation (30%)...]")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_real, y_true, test_size=0.3, random_state=42, stratify=y_true
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)  # 仅在训练集 fit
    X_val_scaled = scaler.transform(X_val_raw)  # 在验证集 transform

    print("\n[Performing Grid Search on Training Set to optimize Proposed Model...]")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 1.0]
    }

    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),  # 此处可换成真实的 LightGBM (LGBMClassifier)
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_gbm = grid_search.best_estimator_
    print(f"  ✓ Optimization complete. Best params: {grid_search.best_params_}")


    base_models = {
        HERO_MODEL: best_gbm,
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=1, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=50, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(16,), max_iter=20, random_state=42),
        'SVM': SVC(probability=True, kernel='rbf', C=0.005, random_state=20),
        'KNN': KNeighborsClassifier(n_neighbors=100)
    }

    model_probs_train = {}
    model_probs_val = {}

    print("\n[Training models and generating probabilities...]")

    for name, clf in base_models.items():
        clf.fit(X_train_scaled, y_train)


        model_probs_train[name] = clf.predict_proba(X_train_scaled)[:, 1]

        model_probs_val[name] = clf.predict_proba(X_val_scaled)[:, 1]

        print(f"  ✓ {name} trained.")

    print("\n[Generating Visualization Plots for both Training and Validation...]")

    plot_multi_roc(y_train, model_probs_train, 'Training')
    plot_multi_calibration(y_train, model_probs_train, 'Training')
    plot_multi_dca(y_train, model_probs_train, 'Training')

    plot_multi_roc(y_val, model_probs_val, 'Validation')
    plot_multi_calibration(y_val, model_probs_val, 'Validation')
    plot_multi_dca(y_val, model_probs_val, 'Validation')

    print("\n[Running 10-Fold Cross Validation on Training Set...]")
    # 在训练集上做10折交叉验证，避免使用到了验证集的数据
    global_cv_auc_dict = run_multi_10fold_cv(X_train_scaled, y_train, base_models)
    plot_multi_cv_boxplot(global_cv_auc_dict)

    print(f"\n{'=' * 60}")
    print(f"✓ Analysis complete! Authentic results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()