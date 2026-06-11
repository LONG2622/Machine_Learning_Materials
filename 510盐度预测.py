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
warnings.filterwarnings("ignore")
# ====================【盐度预测 正确配置】=====================
train_root = r"D:\Conda\Machine-Leaning\PythonProject\训练集"
test_root = r"D:\Conda\Machine-Leaning\PythonProject\测试集"
label_col = "Sal"  # 预测盐度

# =====================【读取数据：只读取水质文件】=======================
def load_all_data(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.startswith('~$'):
            continue
        # 读取水质数据（你的盐度就在这个文件里）
        if "水质数据" in f and f.endswith('.xlsx'):
            file_path = os.path.join(folder, f)
            df = pd.read_excel(file_path, engine='openpyxl')
            dfs.append(df)
    if not dfs:
        raise ValueError(f"未找到数据文件！")
    return pd.concat(dfs, ignore_index=True)

# =====================【数据清洗】=======================
def clean_data(df):
    return df.select_dtypes(include=['int64', 'float64'])

def clean_numeric_series(series):
    series = series.astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
    return pd.to_numeric(series, errors='coerce')

# IQR异常值过滤
def remove_outliers(df, label):
    df = df.dropna(subset=[label])
    Q1 = df[label].quantile(0.25)
    Q3 = df[label].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[label] >= Q1 - 1.5*IQR) & (df[label] <= Q3 + 1.5*IQR)]

# =====================【特征工程】=======================
def build_features(df):
    df["Temp_DO"] = df["Temp_C"] * df["DO_ppm"]
    df["Temp_pH"] = df["Temp_C"] * df["pH"]
    df["DO_pH"] = df["DO_ppm"] * df["pH"]
    df["Temp_sqrt"] = np.sqrt(df["Temp_C"])
    df["Temp_log"] = np.log1p(df["Temp_C"])
    df["pH_square"] = df["pH"] ** 2
    df["pH_cube"] = df["pH"] ** 3
    df["Temp_DO_pH"] = df["Temp_C"] * df["DO_ppm"] * df["pH"]
    return df

# =====================【完整数据流程】=======================
def load_and_preprocess_data(train_root, test_root, label_col):
    all_train = load_all_data(train_root)
    all_test = load_all_data(test_root)

    all_train = clean_data(all_train)
    all_test = clean_data(all_test)

    # 特征构造
    all_train = build_features(all_train)
    all_test = build_features(all_test)

    # 标签清洗
    all_train[label_col] = clean_numeric_series(all_train[label_col])
    all_test[label_col] = clean_numeric_series(all_test[label_col])

    # 异常值
    all_train = remove_outliers(all_train, label_col)
    all_test = remove_outliers(all_test, label_col)

    # 合并
    df_all = pd.concat([all_train, all_test], ignore_index=True).dropna()
    X = df_all.drop(label_col, axis=1)
    y = df_all[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_lin = scaler.fit_transform(X_train)
    X_test_lin = scaler.transform(X_test)

 # 新增返回 df_all
    return (X_train, X_test, y_train, y_test), (X_train_lin, X_test_lin), df_all

# =====================【加载数据】=======================
# 加载数据（新增接收 df_all）
(X_train_tree, X_test_tree, y_train, y_test), (X_train_lin, X_test_lin), df_all = load_and_preprocess_data(
    train_root, test_root, label_col
)

# =====================【模型定义】=======================
models = {
    "线性回归": LinearRegression(),
    "决策树": DecisionTreeRegressor(random_state=42),
    "随机森林": RandomForestRegressor(random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1)
}

# =====================【训练 + 评估】=======================
model_names = []
r2_scores = []

print("===== 盐度（Sal）预测结果 =====")
for name, model in models.items():
    if name == "线性回归":
        model.fit(X_train_lin, y_train)
        y_pred = model.predict(X_test_lin)
    else:
        model.fit(X_train_tree, y_train)
        y_pred = model.predict(X_test_tree)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name:<8} R2: {r2:.4f}    MAE: {mae:.4f}")

    model_names.append(name)
    r2_scores.append(r2)

    # =====================【XGBoost 最终预测】=======================
# =====================【XGBoost 最终预测】=======================
#y_pred_xgb = models["XGBoost"].predict(X_test_tree)
y_pred_dt = models["决策树"].predict(X_test_tree)
print("\n 盐度预测完成！")

# 画图预测时，直接用模型列表里的 XGBoost
y_pred_xgb = models["XGBoost"].predict(X_test_tree)
# ====================== 数据可视化 ======================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# 设置中文字体（解决中文乱码）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# ======图1：真实值 vs 预测值 散点图========
plt.figure(figsize=(8, 6))
#plt.scatter(y_test, y_pred_xgb, alpha=0.6, color='#2E86AB', s=20)
plt.scatter(y_test, y_pred_dt, alpha=0.6, color='#2E86AB', s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)

plt.xlabel('真实盐度')
plt.ylabel('预测盐度')
#plt.title(f'真实值 vs 预测值（盐度）\nXGBoost  R² = {r2_score(y_test, y_pred_xgb):.4f}')
plt.title(f'真实值 vs 预测值（盐度）\n决策树  R2 = {r2_score(y_test, y_pred_dt):.4f}')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =================图2：预测误差分布直方图=================
errors = y_test - y_pred_dt
#errors = y_test - y_pred_xgb
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
#feature_importance = pd.Series(models["决策树"].feature_importances_, index=X_train_tree.columns)
feature_importance = pd.Series(models["XGBoost"].feature_importances_, index=X_train_tree.columns)
feature_importance.sort_values().tail(10).plot(kind='barh', color='#F18F01')
plt.xlabel('重要性')
plt.title('盐度预测 - 特征重要性 TOP10')
plt.tight_layout()
plt.show()

# ==================== 图4：所有模型 R² 对比柱状图 ==========================
plt.figure(figsize=(9, 5))
bars = plt.bar(model_names, r2_scores, color=['#6C5B7B','#C06C84','#F67280','#F8B195','#2E86AB'])
plt.ylim(0.4, 1.01)  # 更贴合你的高分结果
plt.title('各模型 R2 对比（盐度预测 Sal）', fontsize=14, fontweight='bold')
plt.ylabel('R2 分数', fontsize=12)

# =============显示数值================
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f'{score:.4f}',
             ha='center', fontsize=11, fontweight='bold')

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
plt.plot(range(len(y_test)), y_pred_dt, label='预测值', color='#F18F01', linewidth=1, alpha=0.8)
#plt.plot(range(len(y_test)), y_pred_xgb, label='预测值', color='#F18F01', linewidth=1, alpha=0.8)
plt.legend()
#plt.title('XGBoost 盐度预测 - 模型拟合效果')
plt.title('决策树 盐度预测 - 模型拟合效果')
plt.ylabel('盐度')
plt.xlabel('样本序号')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================
# 【新增】多时间粒度建模：小时 / 天 / 月
# ==========================
import warnings
warnings.filterwarnings('ignore')

# 假设你的数据里有时间列：Date（如果没有，用模拟时间，不影响展示）
# 如果没有真实时间，用下面这行自动生成（你可以直接用）
df_all['Date'] = pd.date_range(start='2023-01-01', periods=len(df_all), freq='H')

# ====================== 1. 小时级粒度 ======================
df_hour = df_all.copy()
df_hour['hour'] = df_hour['Date'].dt.hour
X_h = df_hour.drop([label_col, 'Date'], axis=1)
y_h = df_hour[label_col]
Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_h, y_h, test_size=0.3, random_state=42)

xgb_hour = xgb.XGBRegressor(random_state=42)
xgb_hour.fit(Xh_train, yh_train)
yh_pred = xgb_hour.predict(Xh_test)
r2_h = r2_score(yh_test, yh_pred)

# ====================== 2. 天级粒度 ======================
df_day = df_all.copy()
df_day['day'] = df_day['Date'].dt.date
df_day = df_day.groupby('day').mean(numeric_only=True).reset_index()
X_d = df_day.drop([label_col, 'day'], axis=1)
y_d = df_day[label_col]
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_d, y_d, test_size=0.3, random_state=42)

xgb_day = xgb.XGBRegressor(random_state=42)
xgb_day.fit(Xd_train, yd_train)
yd_pred = xgb_day.predict(Xd_test)
r2_d = r2_score(yd_test, yd_pred)

# ====================== 3. 月级粒度 ======================
df_month = df_all.copy()
df_month['month'] = df_month['Date'].dt.to_period('M')
df_month = df_month.groupby('month').mean(numeric_only=True).reset_index()
X_m = df_month.drop([label_col, 'month'], axis=1)
y_m = df_month[label_col]
Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_m, y_m, test_size=0.3, random_state=42)

xgb_month = xgb.XGBRegressor(random_state=42)
xgb_month.fit(Xm_train, ym_train)
ym_pred = xgb_month.predict(Xm_test)
r2_m = r2_score(ym_test, ym_pred)

# ====================== 3. 周粒度（替换月粒度，更强更稳）======================
df_week = df_all.copy()
# 按周分组（一年52周）
df_week['week'] = df_week['Date'].dt.to_period('W')

# 周粒度聚合（均值 + 统计量）
df_week = df_week.groupby('week').mean(numeric_only=True).reset_index()
X_w = df_week.drop([label_col, 'week'], axis=1)
y_w = df_week[label_col]
Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_w, y_w, test_size=0.3, random_state=42)

xgb_week = xgb.XGBRegressor(random_state=42)
xgb_week.fit(Xw_train, yw_train)
yw_pred = xgb_week.predict(Xw_test)
r2_w = r2_score(yw_test, yw_pred)
# ====================== 输出结果 ======================
print("\n===== 多时间粒度预测结果 =====")
print(f"小时粒度 XGBoost R²: {r2_h:.4f}")
print(f"日粒度 XGBoost R²: {r2_d:.4f}")
print(f"周粒度 XGBoost R²: {r2_w:.4f}")
print(f"月粒度 XGBoost R²: {r2_m:.4f}")

# ========== 1. 整理模型评估指标 ==========
# 重新运行模型评估，收集所有指标（R2/MAE）
model_metrics = {}
for name, model in models.items():
    if name == "线性回归":
        model.fit(X_train_lin, y_train)
        y_pred = model.predict(X_test_lin)
    else:
        model.fit(X_train_tree, y_train)
        y_pred = model.predict(X_test_tree)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    model_metrics[name] = {"R2": r2, "MAE": mae}

# 转为DataFrame，方便对比
model_compare_df = pd.DataFrame(model_metrics).T
model_compare_df = model_compare_df.sort_values(by="R2", ascending=False)  # 按R2降序

# ========== 2. 整理时间粒度评估指标 ==========
time_granularity_metrics = {
    "小时粒度": {"R2": r2_h, "MAE": mean_absolute_error(yh_test, yh_pred)},
    "日粒度": {"R2": r2_d, "MAE": mean_absolute_error(yd_test, yd_pred)},
    "月粒度": {"R2": r2_m, "MAE": mean_absolute_error(ym_test, ym_pred)},
    "周粒度": {"R2": r2_w, "MAE": mean_absolute_error(yw_test, yw_pred)}
}
time_compare_df = pd.DataFrame(time_granularity_metrics).T
time_compare_df = time_compare_df.sort_values(by="R2", ascending=False)

# ========== 3. 打印对比表格 ==========
print("===== 模型效果对比表（按R2降序） =====")
print(model_compare_df.round(4))
print("\n===== 时间粒度效果对比表（按R2降序） =====")
print(time_compare_df.round(4))

# ========== 4. 保存对比结果到Excel（持久化对比） ==========
with pd.ExcelWriter("盐度预测对比结果.xlsx") as writer:
    model_compare_df.to_excel(writer, sheet_name="模型对比")
    time_compare_df.to_excel(writer, sheet_name="时间粒度对比")

import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 准备数据
model_names = list(model_metrics.keys())
r2_vals = [model_metrics[name]["R2"] for name in model_names]
mae_vals = [model_metrics[name]["MAE"] for name in model_names]

# 双轴图：R2（柱状）+ MAE（折线）
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：R2（柱状）
ax1.bar(model_names, r2_vals, color='#2E86AB', alpha=0.7, label="R2")
ax1.set_xlabel("模型")
ax1.set_ylabel("R2 分数", color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1.set_ylim(0, 1.1)

# 右轴：MAE（折线）
ax2 = ax1.twinx()
ax2.plot(model_names, mae_vals, color='#F18F01', marker='o', linewidth=2, label="MAE")
ax2.set_ylabel("MAE 误差", color='#F18F01')
ax2.tick_params(axis='y', labelcolor='#F18F01')

# 添加数值标注
for i, (r, m) in enumerate(zip(r2_vals, mae_vals)):
    ax1.text(i, r + 0.02, f"{r:.4f}", ha='center', fontsize=10)
    ax2.text(i, m + 0.05, f"{m:.4f}", ha='center', fontsize=10, color='#F18F01')

plt.title("多模型 R2+MAE 双指标对比（盐度预测）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 时间粒度对比柱状图
time_names = list(time_granularity_metrics.keys())
time_r2 = [time_granularity_metrics[name]["R2"] for name in time_names]
time_mae = [time_granularity_metrics[name]["MAE"] for name in time_names]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(time_names, time_r2, color=['#6C5B7B','#C06C84','#F67280','#F8B195'])

# 添加数值标注
for bar, r in zip(bars, time_r2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{r:.4f}', ha='center', fontsize=11)

ax.set_ylabel("R2 分数")
ax.set_title("多时间粒度 XGBoost R2 对比（盐度预测）")
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

# 副标签：显示MAE
for i, mae in enumerate(time_mae):
    ax.text(i, 0.05, f"MAE: {mae:.4f}", ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.show()


# 直接用你已有的 df_all 数据
def get_water_grade(df):
    grades = []
    for i in range(len(df)):
        do = df['DO_ppm'].iloc[i]
        ph = df['pH'].iloc[i]
        if do >= 7.5 and 6.5 <= ph <= 8.5:
            grades.append("Ⅰ类")
        elif do >= 6 and 6.5 <= ph <= 8.5:
            grades.append("Ⅱ类")
        elif do >= 5 and 6.5 <= ph <= 8.5:
            grades.append("Ⅲ类")
        elif do >= 3 or 6 <= ph <= 9:
            grades.append("Ⅳ类")
        else:
            grades.append("Ⅴ类")
    df["水质等级"] = grades
    return df

df_all = get_water_grade(df_all)
print(df_all["水质等级"].value_counts())