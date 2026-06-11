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
#from sklearn.model_selection import train_test_split

# ======================
# 1. 路径配置
# ======================
train_root = r"D:\Conda\Machine-Leaning\PythonProject\训练集"
test_root = r"D:\Conda\Machine-Leaning\PythonProject\测试集"
label_col = "NO3_Conc"  # 你要预测的目标
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 修正字体（避免乱码）
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 2. 读取数据
# ======================
def load_all_data(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.endswith('水质数据.xlsx'):
            file_path = os.path.join(folder, f)
            df = pd.read_excel(file_path, engine='openpyxl')
            dfs.append(df)
    if not dfs:
        raise ValueError(f"文件夹{folder}中未找到xlsx文件！")
    return pd.concat(dfs, ignore_index=True)

all_train = load_all_data(train_root)
all_test = load_all_data(test_root)

# ======================
# 3. 只保留数字列 + 标签（自动删掉站点名、时间等文字列）
# ======================
def clean_data(df, label_col):
    # 保留数字列
    df_num = df.select_dtypes(include=['int64', 'float64'])
    # 把标签加回来（防止被删掉）
    if label_col in df.columns and label_col not in df_num.columns:
        df_num[label_col] = df[label_col]
    return df_num

all_train = clean_data(all_train, label_col)
all_test = clean_data(all_test, label_col)

# ======================
# 4. 清洗标签（去掉 mg/L、L 等单位）
# ======================
def clean_numeric_series(series):
    series = series.astype(str)
    series = series.str.extract(r'(\d+\.?\d*)', expand=False)
    series = pd.to_numeric(series, errors='coerce')
    return series

all_train[label_col] = clean_numeric_series(all_train[label_col])
all_test[label_col] = clean_numeric_series(all_test[label_col])

# ======================
# 关键：不填充！直接删除有空值的行（只保留完整数据）
# ======================
all_train = all_train.dropna()
all_test = all_test.dropna()

# ======================
# 5. 分离 X 和 y
# ======================
X_train = all_train.drop(label_col, axis=1)
y_train = all_train[label_col]
X_test = all_test.drop(label_col, axis=1)
y_test = all_test[label_col]

# ======================
# 6. 训练模型
# ======================
model_lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])#线性回归
#决策树
model_dt = Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeRegressor(random_state=42, max_depth=10))])
#随机森林
model_rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)

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