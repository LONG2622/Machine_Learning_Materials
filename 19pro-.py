import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

# ======================================
# 配置
# ======================================
train_root = r"D:\Conda\Machine-Leaning\PythonProject\训练集"
test_root = r"D:\Conda\Machine-Leaning\PythonProject\测试集"
label_col = "NO3_Conc"

# ======================================
# 读取数据
# ======================================
def load_all_data(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.startswith('~$'): continue
        if f.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(folder, f), engine='openpyxl')
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

all_train = load_all_data(train_root)
all_test = load_all_data(test_root)

# ======================================
# 【关键修复】清洗 NO3 单位（你之前就是靠这行解决的！）
# ======================================
def clean_no3(series):
    series = series.astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
    return pd.to_numeric(series, errors='coerce')

if label_col in all_train.columns:
    all_train[label_col] = clean_no3(all_train[label_col])
if label_col in all_test.columns:
    all_test[label_col] = clean_no3(all_test[label_col])

# ======================================
#  现在再保留数字列（NO3 不会丢了！）
# ======================================
all_train = all_train.select_dtypes(include=['int64', 'float64'])
all_test = all_test.select_dtypes(include=['int64', 'float64'])

# ======================================
# 【正确位置】加交互特征！！！
# ======================================
all_train["Temp_DO"] = all_train["Temp_C"] * all_train["DO_ppm"]
all_train["Temp_pH"] = all_train["Temp_C"] * all_train["pH"]
all_test["Temp_DO"]  = all_test["Temp_C"] * all_test["DO_ppm"]
all_test["Temp_pH"]  = all_test["Temp_C"] * all_test["pH"]

# ======================================
# 【正确位置】异常值处理！！！
# ======================================
def remove_outliers(df, label):
    m = df[label].mean()
    s = df[label].std()
    return df[(df[label] > m-3*s) & (df[label] < m+3*s)]

all_train = remove_outliers(all_train, label_col)
all_test = remove_outliers(all_test, label_col)

# ======================================
# 空值处理
# ======================================
all_train = all_train.dropna()
all_test = all_test.dropna()

# ======================================
# 划分 X y
# ======================================
X_train = all_train.drop(label_col, axis=1)
y_train = all_train[label_col]
X_test  = all_test.drop(label_col, axis=1)
y_test  = all_test[label_col]

# ======================================
# 模型（完全优化版）
# ======================================
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42, max_depth=10),
    "随机森林": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=42
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=6, num_leaves=24,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.2,
        random_state=42, verbosity=-1
    )
}

# ======================================
# 训练评估
# ======================================
print("===== 模型对比结果 =====")
best_r2 = -1
best_name = ""


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