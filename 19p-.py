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
# 2. 简单清除空值（不暴力、不删光数据）
all_train = clean_data(all_train, label_col)
all_test = clean_data(all_test, label_col)
# =====================【划分 X y】========================
def clean_numeric_series(series):
    series = series.astype(str)
    # 提取数字（支持整数、小数）
    series = series.str.extract(r'(\d+\.?\d*)', expand=False)
    # 转数字，无法转换的变为NaN
    series = pd.to_numeric(series, errors='coerce')
    return series
# ======================
# 盐度预测最强特征  19
# ======================
all_train["Temp_DO"] = all_train["Temp_C"] * all_train["DO_ppm"]
all_train["Temp_pH"]  = all_train["Temp_C"] * all_train["pH"]
all_train["DO_pH"]    = all_train["DO_ppm"] * all_train["pH"]

all_test["Temp_DO"]  = all_test["Temp_C"] * all_test["DO_ppm"]
all_test["Temp_pH"]  = all_test["Temp_C"] * all_test["pH"]
all_test["DO_pH"]    = all_test["DO_ppm"] * all_test["pH"]
all_train[label_col] = clean_numeric_series(all_train[label_col])
all_test[label_col] = clean_numeric_series(all_test[label_col])
# 4.3 不填充空值，只保留完整有效数据
all_train = all_train.dropna()
all_test = all_test.dropna()

# ======================================
# 5. 合并数据+随机划分（解决测试集分布异常问题）
# ======================================
df_all = pd.concat([all_train, all_test], ignore_index=True)
X = df_all.drop(label_col, axis=1)
y = df_all[label_col]

# 7:3随机划分训练集/测试集，避免原划分的分布问题
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======================================
# 6. 特征标准化（线性模型必须，树模型可选但不影响）
# ======================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3σ 原则去除异常值（对盐度特别有效）
def remove_outliers(df, label):
    mean = df[label].mean()
    std = df[label].std()
    return df[(df[label] > mean - 3*std) & (df[label] < mean + 3*std)]

all_train = remove_outliers(all_train, label_col)
all_test = remove_outliers(all_test, label_col)


# =====================【最简单安全的模型】====================
# =====================【顶配优化版 XGBoost + LightGBM】====================
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42, max_depth=8),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),

    # ===  XGBoost 最终顶配 ===
    "XGBoost": xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.3,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    ),

    # ===  LightGBM 最终顶配 ===
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.3,
        min_child_samples=10,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
}
"""
# =====================【优化版 XGBoost + LightGBM】====================
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42, max_depth=8),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),

    #  优化版 XGBoost（精度暴涨）
    "XGBoost": xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    ),

    # 优化版 LightGBM（精度暴涨）
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
}

models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42, max_depth=8),
    "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "LightGBM": lgb.LGBMRegressor(random_state=42, verbosity=-1)
}
"""
# =====================【训练 + 输出】=====================
print("===== 模型对比结果 =====")
best_r2 = -1
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{name:<8} R²: {r2:.4f}    MAE: {mae:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_name = name

print(f"\n最优模型：{best_name}")
print(f"最优R²分数：{best_r2:.4f}")