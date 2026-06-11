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
# ======================================
# 1. 路径配置（完全匹配你的文件夹结构）
# ======================================
train_root = r"D:\Conda\Machine-Leaning\PythonProject\训练集"
test_root = r"D:\Conda\Machine-Leaning\PythonProject\测试集"
label_col = "NO3_Conc"  # 预测目标：硝酸盐浓度

# ======================================
# 2. 数据读取函数（自动跳过临时文件，读取Excel）
# ======================================
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


# ======================================
# 3. 读取训练集+测试集
# ======================================
all_train = load_all_data(train_root)
all_test = load_all_data(test_root)

# ======================================
# 4. 数据清洗（只保留数字列+标签，处理单位）
# ======================================
# 4.1 保留数字列，自动过滤站点名、时间等文字列
def clean_data(df, label_col):
    df_num = df.select_dtypes(include=['int64', 'float64'])
    # 确保标签列不被误删
    if label_col in df.columns and label_col not in df_num.columns:
        df_num[label_col] = df[label_col]
    return df_num

all_train = clean_data(all_train, label_col)
all_test = clean_data(all_test, label_col)

# 4.2 清洗标签列（提取数字，去除mg/L、L等单位）
def clean_numeric_series(series):
    series = series.astype(str)
    # 提取数字（支持整数、小数）
    series = series.str.extract(r'(\d+\.?\d*)', expand=False)
    # 转数字，无法转换的变为NaN
    series = pd.to_numeric(series, errors='coerce')
    return series

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

# ======================
# 盐度预测最强特征  19
# ======================
all_train["Temp_DO"] = all_train["Temp_C"] * all_train["DO_ppm"]
all_train["Temp_pH"]  = all_train["Temp_C"] * all_train["pH"]
all_train["DO_pH"]    = all_train["DO_ppm"] * all_train["pH"]

all_test["Temp_DO"]  = all_test["Temp_C"] * all_test["DO_ppm"]
all_test["Temp_pH"]  = all_test["Temp_C"] * all_test["pH"]
all_test["DO_pH"]    = all_test["DO_ppm"] * all_test["pH"]

# 3σ 原则去除异常值（对盐度特别有效）
def remove_outliers(df, label):
    mean = df[label].mean()
    std = df[label].std()
    return df[(df[label] > mean - 3*std) & (df[label] < mean + 3*std)]

all_train = remove_outliers(all_train, label_col)
all_test = remove_outliers(all_test, label_col)

# ======================================
# 7. 模型训练与评估
# ======================================
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42, max_depth=10),
    "随机森林": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    #"LightGBM": lgb.LGBMRegressor(n_estimators=300, random_state=42, verbosity=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=500, random_state=42),
    #修改light19
    "LightGBM":lgb.LGBMRegressor(
    n_estimators = 1000,      # 更多树，学习更细
    learning_rate = 0.02,     # 小学习率，更稳定
    max_depth = 10,            # 不深不浅，不过拟合
    num_leaves = 30,         # 盐度数据最舒服的叶子数
    subsample=0.8,         # 随机采样数据，抗噪
    colsample_bytree=0.8,  # 随机采样特征，增强泛化
    reg_alpha=0.1,          # 正则化，防止过拟合
    reg_lambda=0.2,
    random_state=42,
    verbosity=-1
)
}

print("===== 模型对比结果 =====")
best_score = -np.inf
best_name = ""
best_r2 = -1
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{name:<6} R²: {r2:.4f}    MAE: {mae:.4f}")

    if r2 > best_score:
        best_score = r2
        best_name = name

print(f"\n最优模型：{best_name}")
print(f"最优R²分数：{best_score:.4f}")
print("\n全部运行成功！")

#数据可视化处理
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10,6))

pred_lr = models["线性回归"].predict(X_test)
pred_lgb = models["LightGBM"].predict(X_test)
pred_rf = models["随机森林"].predict(X_test)
pred_dt = models["决策树"].predict(X_test)
pred_xgb   = models["XGBoost"].predict(X_test)

plt.plot(y_test.values, label='真实值 NO3_Conc', c='black', linewidth=2)
plt.plot(pred_lgb, label='LightGBM 预测', c='#2E8B57', alpha=0.8)
plt.plot(pred_rf, label='随机森林 预测', c='#FF6347', alpha=0.7)
plt.plot(pred_xgb, label='XGBoost 预测', c='#1E90FF', alpha=0.7)

plt.title('真实值 vs 预测值对比（最优模型）', fontsize=14)
plt.xlabel('测试集样本序号')
plt.ylabel('NO3_Conc 浓度')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

#特征重要性排名
# 特征名（你之前的列，我帮你复原）
feature_names = all_train.drop(label_col, axis=1).columns.tolist()

# 特征重要性
importances = models["LightGBM"].feature_importances_

plt.figure(figsize=(10,5))
plt.title('LightGBM 特征重要性（影响 NO3_Conc 程度）', fontsize=14)
plt.xlabel('重要性分数')
plt.gca().invert_yaxis()  # 最重要的放最上面
plt.grid(alpha=0.3)
plt.show()