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
# =====================【修复后：数据清洗】=======================
def clean_data(df):
    # 1. 先把营养盐列强制转成数字（处理 ######## 异常值）
    for col in ["NO3_Conc", "NO2_Conc", "NH3_Conc", "PO4_Conc"]:
        if col in df.columns:
            df[col] = clean_numeric_series(df[col])

    # 2. 再保留数字列（现在营养盐已经是数字了，不会被删）
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
    return df[(df[label] >= Q1 - 1.5 * IQR) & (df[label] <= Q3 + 1.5 * IQR)]

"""
def clean_data(df, label_cols=None):
    if label_cols is None:
        label_cols = []

    # 先把营养盐列转为数值型（处理########这类异常值）
    for col in ["PO4_Conc", "NH3_Conc", "NO2_Conc", "NO3_Conc"]:
        if col in df.columns:
            # 用clean_numeric_series把字符串/异常值转为数值
            df[col] = clean_numeric_series(df[col])

    # 再筛选数值型列，同时保留指定的label_cols
    df_num = df.select_dtypes(include=['int64', 'float64'])
    for col in label_cols:
        if col in df.columns and col not in df_num.columns:
            df_num[col] = df[col]

    return df_num

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
    return df[(df[label] >= Q1 - 1.5 * IQR) & (df[label] <= Q3 + 1.5 * IQR)]"""


# =====================【特征工程】=======================
def build_features(df):
    # 先检查列是否存在，避免KeyError
    required_cols = ["Temp_C", "DO_ppm", "pH"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺失关键列：{col}")

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

    # 合并（返回完整的df_all，解决作用域问题）
    df_all = pd.concat([all_train, all_test], ignore_index=True).dropna()
    X = df_all.drop(label_col, axis=1)
    y = df_all[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train_lin = scaler.fit_transform(X_train)
    X_test_lin = scaler.transform(X_test)

    return (X_train, X_test, y_train, y_test), (X_train_lin, X_test_lin), df_all  # 新增返回df_all


# =====================【加载数据】=======================
# 接收返回的df_all
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
y_pred_xgb = models["XGBoost"].predict(X_test_tree)
print("\n 盐度预测完成！")

# ==================== K-Means 水质聚类（修改核心区）===================
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 设置中文字体（解决聚类可视化中文乱码）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def water_quality_clustering(df_all, cluster_cols=None):
    """
    水质聚类函数
    :param df_all: 预处理后的完整数据
    :param cluster_cols: 聚类特征列，默认使用核心水质指标
    :return: 包含聚类结果的df、KMeans模型

    print("你数据里所有的列名：")
    print(df_all.columns.tolist())
    """
    # 1. 定义聚类特征（适配你的数据列名，若NO3_Conc不存在则替换/删除）
    if cluster_cols is None:
        cluster_cols = ["Temp_C", "pH", "DO_ppm", "Sal",  "NO3_Conc", "PO4_Conc", "NO2_Conc","NH3_Conc"]
        #cluster_cols = ["Temp_C", "pH", "DO_ppm", "Sal", "NO3_Conc"]  # 移除NO3_Conc（若数据中无此列）
        # 可选：如果有硝酸盐列，替换为实际列名，例如：
        # cluster_cols = ["Temp_C", "pH", "DO_ppm", "Sal", "硝酸盐"]

    # 检查聚类特征是否存在
    missing_cols = [col for col in cluster_cols if col not in df_all.columns]
    if missing_cols:
        raise ValueError(f"聚类特征列缺失：{missing_cols}，请检查数据列名！")

    # 2. 数据预处理（空值处理+标准化）
    df_cluster = df_all[cluster_cols].copy()
    df_cluster = df_cluster.dropna()  # 聚类前删除空值
    if len(df_cluster) == 0:
        raise ValueError("聚类数据为空！请检查数据清洗逻辑")

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_cluster)

    # 3. 肘部法则自动选择最优K值（替代固定K=3）
    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init消除警告
        km.fit(X_cluster)
        inertia.append(km.inertia_)

    # 绘制肘部法则图
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, "o-", color="#2E86AB")
    plt.title("K-Means 肘部法则图（选择最优聚类数）")
    plt.xlabel("聚类数量 K")
    plt.ylabel("损失值（Inertia）")
    plt.grid(alpha=0.3)
    plt.show()

    # 4. 选择K值（可根据肘部图手动调整，默认3）
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_cluster["cluster"] = kmeans.fit_predict(X_cluster)

    # 5. 合并聚类结果到原始数据
    df_all = df_all.merge(df_cluster["cluster"], left_index=True, right_index=True, how="left")

    # 6. 输出聚类统计结果
    print("\n===== K-Means 聚类结果（每类均值）=====")
    cluster_stats = df_cluster.groupby("cluster")[cluster_cols].mean()
    print(cluster_stats)

    # 7. TSNE降维可视化
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_cluster)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df_cluster["cluster"],
                          cmap="viridis", s=15, alpha=0.8)
    plt.title(f"K-Means 水质聚类可视化（K={optimal_k}）")
    plt.xlabel("TSNE 维度1")
    plt.ylabel("TSNE 维度2")
    plt.colorbar(scatter, label="Cluster 类别")
    plt.grid(alpha=0.3)
    plt.show()

    return df_all, kmeans
# 执行聚类
df_all_with_cluster, kmeans_model = water_quality_clustering(df_all)
# 可选：查看每个聚类的样本数量
print("\n===== 各聚类样本数量 =====")
print(df_all_with_cluster["cluster"].value_counts())