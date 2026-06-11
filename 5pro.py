import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ======================
# 1. 路径与参数配置
# ======================
train_root = r"D:\Conda\Machine-Leaning\PythonProject\训练集"
test_root = r"D:\Conda\Machine-Leaning\PythonProject\测试集"
label_col = "Sal"  # 预测目标列
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 修正字体（避免乱码）
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 2. 数据加载（保留数字类型）
# ======================
def load_all_data(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.endswith('.xlsx'):
            file_path = os.path.join(folder, f)
            df = pd.read_excel(file_path, engine='openpyxl')
            dfs.append(df)
    if not dfs:
        raise ValueError(f"文件夹{folder}中未找到xlsx文件！")
    return pd.concat(dfs, ignore_index=True)

all_train = load_all_data(train_root)
all_test = load_all_data(test_root)

# 仅保留数值型特征（避免非数值列干扰）
all_train = all_train.select_dtypes(include=['int64', 'float64'])
all_test = all_test.select_dtypes(include=['int64', 'float64'])

# 检查标签列是否存在
if label_col not in all_train.columns or label_col not in all_test.columns:
    raise ValueError(f"标签列{label_col}不在数据集中！")

# ======================
# 3. 特征/标签分离（仅做1次空值填充）
# ======================
# 分离X和y
X_train = all_train.drop(label_col, axis=1)
y_train = all_train[label_col]
X_test = all_test.drop(label_col, axis=1)
y_test = all_test[label_col]

# 定义统一的空值填充器（仅1次，避免重复）
imputer = SimpleImputer(strategy='median')
# 填充特征空值（fit只在训练集做，避免数据泄露）
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
# 填充标签空值
y_train = y_train.fillna(y_train.median())
y_test = y_test.fillna(y_test.median())

# ======================
# 4. 模型定义（统一Pipeline格式，保证预处理一致）
# ======================
# 4.1 线性回归（需要标准化）
model_lr = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),  # 兜底填充（防止漏网之鱼）
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# 4.2 决策树（不需要标准化，仅保留imputer）
model_dt = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("model", DecisionTreeRegressor(random_state=42, max_depth=10))
])

# 4.3 随机森林（不需要标准化，仅保留imputer）
model_rf = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42))
])

# ======================
# 5. 模型训练与预测
# ======================
# 训练
model_lr.fit(X_train, y_train)
model_dt.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# 预测
pred_lr = model_lr.predict(X_test)
pred_dt = model_dt.predict(X_test)
pred_rf = model_rf.predict(X_test)

# ======================
# 6. 模型评估（统一指标）
# ======================
# 计算R²和MAE
metrics = {
    "线性回归": {
        "R²": r2_score(y_test, pred_lr),
        "MAE": mean_absolute_error(y_test, pred_lr)
    },
    "决策树": {
        "R²": r2_score(y_test, pred_dt),
        "MAE": mean_absolute_error(y_test, pred_dt)
    },
    "随机森林": {
        "R²": r2_score(y_test, pred_rf),
        "MAE": mean_absolute_error(y_test, pred_rf)
    }
}

# 输出评估结果
print("===== 模型对比结果 =====")
for name, metric in metrics.items():
    print(f"{name:8} R²: {metric['R²']:.4f}    MAE: {metric['MAE']:.4f}")

# 找出最优模型（按R²排序）
best_model = max(metrics.keys(), key=lambda x: metrics[x]["R²"])
print(f"\n最优模型：{best_model}")
print(f"最优R²分数：{metrics[best_model]['R²']:.4f}")

# ======================
# 7. 可视化（修正路径+合理展示）
# ======================
plt.figure(figsize=(12, 6))
# 绘制真实值与各模型预测值的对比
plt.scatter(y_test, pred_lr, color='blue', alpha=0.5, label='线性回归', s=10)
plt.scatter(y_test, pred_dt, color='red', alpha=0.5, label='决策树', s=10)
plt.scatter(y_test, pred_rf, color='green', alpha=0.5, label='随机森林', s=10)
# 绘制y=x参考线（完美预测）
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='完美预测线')

plt.xlabel('真实值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title(f'{label_col} 真实值与预测值对比', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
# 修正保存路径（正确解析变量+合法文件名）
plt.savefig(f'{label_col}_真实值与预测值对比.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n全部运行成功！")