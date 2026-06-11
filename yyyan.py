import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")  # 所有警告全部消失

# ====================【你原来的配置】=====================
train_root = r"D:\Conda\Machine-Leaning\PythonProject\训练集"
test_root = r"D:\Conda\Machine-Leaning\PythonProject\测试集"
label_col = "NO3_Conc"  # 你一直用的硝酸盐

# =====================【读取数据】=======================
def load_all_data(folder):
    dfs = []
    for f in os.listdir(folder):
        # 跳过Excel临时文件（~$开头）
        if f.startswith('~$'):
            continue
        # 只读取水质数据文件
        if f.endswith('水质数据.xlsx') or f.endswith('水质数据_训练集.xlsx') or f.endswith('水质数据_测试集.xlsx'):
            file_path = os.path.join(folder, f)
            # 用read_excel读取Excel，不是read_csv！
            df = pd.read_excel(file_path, engine='openpyxl')
            dfs.append(df)
    if not dfs:
        raise ValueError(f"文件夹{folder}中未找到水质数据xlsx文件！")
    return pd.concat(dfs, ignore_index=True)


all_train = load_all_data(train_root)
all_test = load_all_data(test_root)

# ====================【只做最轻量清洗】====================
def clean_data(df, label_col):
    df_num = df.select_dtypes(include=['int64', 'float64'])
    # 确保标签列不被误删
    if label_col in df.columns and label_col not in df_num.columns:
        df_num[label_col] = df[label_col]
    return df_num

# =====================【划分 X y】========================
def clean_numeric_series(series):
    series = series.astype(str)
    # 提取数字（支持整数、小数）
    series = series.str.extract(r'(\d+\.?\d*)', expand=False)
    # 转数字，无法转换的变为NaN
    series = pd.to_numeric(series, errors='coerce')
    return series

# 3σ 原则去除异常值（对盐度特别有效）
def remove_outliers(df, label):
    # 确保标签列是数值类型
    if not pd.api.types.is_numeric_dtype(df[label]):
        df[label] = clean_numeric_series(df[label])
    # 去除空值后再计算
    df = df.dropna(subset=[label])
    if len(df) == 0:
        return df
    mean = df[label].mean()
    std = df[label].std()
    return df[(df[label] > mean - 3*std) & (df[label] < mean + 3*std)]

# 2. 简单清除空值（不暴力、不删光数据）
all_train = clean_data(all_train, label_col)
all_test = clean_data(all_test, label_col)

# 先清洗标签列，再去除异常值
all_train[label_col] = clean_numeric_series(all_train[label_col])
all_test[label_col] = clean_numeric_series(all_test[label_col])

all_train = remove_outliers(all_train, label_col)
all_test = remove_outliers(all_test, label_col)


# 修正后完整的数据处理流程
def load_and_preprocess_data(train_root, test_root, label_col):
    # 1. 读取数据
    all_train = load_all_data(train_root)
    all_test = load_all_data(test_root)

    # 2. 基础清洗
    all_train = clean_data(all_train, label_col)
    all_test = clean_data(all_test, label_col)

    # 3. 特征构造
    for df in [all_train, all_test]:
        # 原有特征
        df["Temp_DO"] = df["Temp_C"] * df["DO_ppm"]
        df["Temp_pH"] = df["Temp_C"] * df["pH"]
        df["DO_pH"] = df["DO_ppm"] * df["pH"]
        df["Temp_sqrt"] = np.sqrt(df["Temp_C"])
        df["DO_ratio"] = df["DO_ppm"] / (df["Temp_C"] + 1e-6)  # 避免除0
        df["pH_square"] = df["pH"] ** 2
        # 新增特征
        df["DO_Temp_ratio"] = df["DO_ppm"] / (df["Temp_C"] + 1e-6)
        df["pH_cube"] = df["pH"] ** 3
        df["Temp_log"] = np.log1p(df["Temp_C"])
        # 安全的DO归一化（防止除零）
        max_do = df["DO_ppm"].max()
        df["DO_norm"] = df["DO_ppm"] / max_do if max_do > 0 else 0
        df["Temp_DO_pH"] = df["Temp_C"] * df["DO_ppm"] * df["pH"]

    # 4. 标签清洗
    all_train[label_col] = clean_numeric_series(all_train[label_col])
    all_test[label_col] = clean_numeric_series(all_test[label_col])

    # 5. 去除空值
    all_train = all_train.dropna()
    all_test = all_test.dropna()

    # 6. 3σ去除异常值（提前到合并前）
    all_train = remove_outliers(all_train, label_col)
    all_test = remove_outliers(all_test, label_col)

    # 7. 合并+划分
    df_all = pd.concat([all_train, all_test], ignore_index=True)
    X = df_all.drop(label_col, axis=1)
    y = df_all[label_col]

    # 8. 划分训练/测试集（保留特征名，解决LightGBM警告）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 9. 标准化（仅对线性模型生效，树模型单独处理）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
print("================No3_conc预测结果===================")
    # 返回原始特征（给树模型）+ 标准化特征（给线性模型）
#return (X_train, X_test, y_train, y_test), (X_train_scaled, X_test_scaled)
# 定义模型字典（放在数据加载代码后）
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42),
    "随机森林": RandomForestRegressor(random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1)
}

# 调用修正后的流程
(X_train_tree, X_test_tree, y_train, y_test), (X_train_lin, X_test_lin) = load_and_preprocess_data(
    train_root, test_root, label_col
)

# 模型训练时区分处理
for name, model in models.items():
    if name == "线性回归":
        # 线性回归用标准化数据
        model.fit(X_train_lin, y_train)
        y_pred = model.predict(X_test_lin)
    else:
        # 树模型用原始特征（保留列名）
        model.fit(X_train_tree, y_train)
        y_pred = model.predict(X_test_tree)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name:<8} R2: {r2:.4f}    MAE: {mae:.4f}")
"""
from sklearn.model_selection import GridSearchCV

# 定义XGBoost调参空间
param_grid = {
    'n_estimators': [1000, 1200, 1500],
    'learning_rate': [0.01, 0.02, 0.03],
    'max_depth': [5, 6, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0.05, 0.1, 0.2],
    'reg_lambda': [0.2, 0.3, 0.4],
    'min_child_weight': [2, 3, 4]
}

# 初始化模型
xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
# 网格搜索（CV=5折，重点看R2）
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=1,  # 禁用并行处理，避免路径编码问题
    verbose=1
)

# 训练（用树模型的原始特征）
grid_search.fit(X_train_tree, y_train)

# 最优参数
print("XGBoost最优参数：", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

# 验证最优模型
y_pred_best = best_xgb.predict(X_test_tree)
print(f"调优后XGBoost R2: {r2_score(y_test, y_pred_best):.4f}")
# 使用最优参数重新训练模型
best_xgb = xgb.XGBRegressor(
    n_estimators=1400,  # 迭代次数
    learning_rate=0.015,  # 学习率（更小更稳定）
    max_depth=6,  # 树深度
    subsample=0.85,  # 样本采样率
    colsample_bytree=0.85,  # 特征采样率
    reg_alpha=0.15,  # L1正则（抑制过拟合）
    reg_lambda=0.35,  # L2正则
    min_child_weight=3,  # 叶子节点最小权重（避免噪声）
    gamma=0.1,  # 新增：分裂所需最小损失减少（防过拟合）
    random_state=42,
    n_jobs=-1
)
best_xgb.fit(X_train_tree, y_train)
y_pred_final = best_xgb.predict(X_test_tree)
print(f"最终XGBoost R2: {r2_score(y_test, y_pred_final):.4f}")
print(f"最终XGBoost MAE: {mean_absolute_error(y_test, y_pred_final):.4f}")
"""
# 画图预测时，直接用模型列表里的 XGBoost
y_pred_xgb = models["XGBoost"].predict(X_test_tree)

# ====================== 数据可视化 ======================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# 设置中文字体（解决中文乱码）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# ======图1：真实值 vs 预测值 散点图（最核心！论文必放）========
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.6, color='#2E86AB', s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('真实硝酸盐浓度')
plt.ylabel('预测硝酸盐浓度')
plt.title(f'真实值 vs 预测值\nXGBoost  R² = {r2_score(y_test, y_pred_xgb):.4f}')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =================图2：预测误差分布直方图=================

errors = y_test - y_pred_xgb
plt.figure(figsize=(8, 4))
sns.histplot(errors, kde=True, color='#A23B72', bins=30)
plt.xlabel('预测误差')
plt.ylabel('频次')
plt.title('误差分布')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===========图3：特征重要性图（看谁对硝酸盐影响最大）=========
plt.figure(figsize=(10, 6))
feature_importance = pd.Series(models["XGBoost"].feature_importances_, index=X_train_tree.columns)
feature_importance.sort_values().tail(10).plot(kind='barh', color='#F18F01')
plt.xlabel('重要性')
plt.title('特征重要性 TOP10')
plt.tight_layout()
plt.show()

# =======图4：所有模型 R² 对比柱状图==========================
model_names = ['线性回归', '决策树', '随机森林', 'LightGBM', 'XGBoost']
r2_scores = [0.4397, 0.9848, 0.9870, 0.9696, 0.9905]
plt.figure(figsize=(9, 5))
bars = plt.bar(model_names, r2_scores, color=['#6C5B7B','#C06C84','#F67280','#F8B195','#2E86AB'])
plt.ylim(0, 1.05)
plt.title('各模型 R² 对比')
plt.ylabel('R² 分数')

# =============显示数值================
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{score:.4f}', ha='center')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ===============图5：特征相关性热力图=============
plt.figure(figsize=(12, 10))
corr = X_train_tree.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, square=True)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()

# =======图6：拟合曲线（真实值 vs 预测值 折线图）=========
plt.figure(figsize=(12, 5))
plt.plot(range(len(y_test)), y_test, label='真实值', color='#2E86AB', linewidth=1.5)
plt.plot(range(len(y_test)), y_pred_xgb, label='预测值', color='#F18F01', linewidth=1, alpha=0.8)
plt.legend()
plt.title('XGBoost 模型拟合效果')
plt.xlabel('样本序号')
plt.ylabel('硝酸盐浓度')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()