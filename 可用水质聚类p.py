import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score
import lightgbm as lgb
import xgboost as xgb
import warnings
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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


# IQR异常值过滤（用于盐度预测的标签清洗）
def remove_outliers(df, label):
    df = df.dropna(subset=[label])
    Q1 = df[label].quantile(0.25)
    Q3 = df[label].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[label] >= Q1 - 1.5 * IQR) & (df[label] <= Q3 + 1.5 * IQR)]


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

# ==================== K-Means 水质聚类（核心优化版）===================
# 设置中文字体（解决聚类可视化中文乱码）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# 优化点1：鲁棒的聚类预处理（替换原简单删除空值+StandardScaler）
def preprocess_for_clustering(df, cluster_cols):
    """聚类专用预处理：鲁棒标准化+异常值处理+空值填充"""
    df_cluster = df[cluster_cols].copy()

    # 1. 空值填充（用中位数更鲁棒）
    df_cluster = df_cluster.fillna(df_cluster.median())

    # 2. 异常值处理（IQR替换边界值，保留样本）
    for col in df_cluster.columns:
        Q1 = df_cluster[col].quantile(0.25)
        Q3 = df_cluster[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cluster[col] = np.where(df_cluster[col] < lower_bound, lower_bound, df_cluster[col])
        df_cluster[col] = np.where(df_cluster[col] > upper_bound, upper_bound, df_cluster[col])

    # 3. 鲁棒标准化（抗异常值）
    scaler = RobustScaler()
    X_cluster = scaler.fit_transform(df_cluster)

    return X_cluster, df_cluster, scaler


# 优化点2：自动选K（肘部法则+轮廓系数）
def find_optimal_k(X_cluster, max_k=10):
    """自动选择最优K值：肘部法则+轮廓系数"""
    inertia = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)  # 轮廓系数至少需要2个聚类

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = km.fit_predict(X_cluster)
        inertia.append(km.inertia_)
        silhouette_avg = silhouette_score(X_cluster, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # 可视化肘部法则+轮廓系数
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 肘部图
    ax1.plot(K_range, inertia, "o-", color="#2E86AB")
    ax1.set_title("肘部法则（损失值）")
    ax1.set_xlabel("K值")
    ax1.set_ylabel("Inertia")
    ax1.grid(alpha=0.3)

    # 轮廓系数图
    ax2.plot(K_range, silhouette_scores, "o-", color="#A23B72")
    ax2.set_title("轮廓系数（Silhouette Score）")
    ax2.set_xlabel("K值")
    ax2.set_ylabel("轮廓系数")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 选择轮廓系数最大的K值
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n基于轮廓系数的最优K值：{optimal_k}")
    return optimal_k


# 优化点3：聚类结果可解释性（业务标签+特征占比）
def interpret_clusters(df_cluster, cluster_cols):
    """聚类结果解释：计算每类特征的相对占比+业务标签"""
    cluster_stats = df_cluster.groupby("cluster")[cluster_cols].mean()

    # 1. 特征相对占比（相对于整体均值）
    overall_mean = df_cluster[cluster_cols].mean()
    cluster_ratio = cluster_stats / overall_mean
    print("\n===== 聚类特征相对占比（相对于整体均值）=====")
    print(cluster_ratio.round(2))  # >1表示高于平均，<1低于平均

    # 2. 自动生成业务标签（盐度+溶解氧维度）
    cluster_labels = {}
    for cluster_id in cluster_stats.index:
        sal_level = "高盐度" if cluster_stats.loc[cluster_id, "Sal"] > overall_mean["Sal"] else "低盐度"
        do_level = "高溶解氧" if cluster_stats.loc[cluster_id, "DO_ppm"] > overall_mean["DO_ppm"] else "低溶解氧"
        cluster_labels[cluster_id] = f"{sal_level}-{do_level}组"

    print("\n===== 聚类业务标签 =====")
    for cid, label in cluster_labels.items():
        print(f"聚类{cid}：{label}")

    return cluster_stats, cluster_labels

# 优化点4：可视化升级（雷达图+饼图）
def plot_cluster_radar(cluster_stats, cluster_cols, cluster_labels):
    """雷达图展示每类特征均值"""
    # 标准化特征（0-1），方便对比
    stats_scaled = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())

    # 雷达图角度
    angles = np.linspace(0, 2 * np.pi, len(cluster_cols), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for cluster_id in stats_scaled.index:
        values = stats_scaled.loc[cluster_id].tolist()
        values += values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=cluster_labels.get(cluster_id, f"聚类{cluster_id}"))
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cluster_cols)
    ax.set_title("各聚类特征雷达图（标准化）", size=15)
    ax.legend(loc="upper right")
    plt.show()

# 主聚类函数（整合所有优化）
def water_quality_clustering(df_all, cluster_cols=None):
    """
    水质聚类函数（优化版）
    :param df_all: 预处理后的完整数据
    :param cluster_cols: 聚类特征列
    :return: 包含聚类结果的df、KMeans模型
    """
    print("\n你数据里所有的列名：")
    print(df_all.columns.tolist())

    # 1. 定义聚类特征
    if cluster_cols is None:
        cluster_cols = ["Temp_C", "pH", "DO_ppm", "Sal", "NO3_Conc", "PO4_Conc", "NO2_Conc", "NH3_Conc"]

    # 检查聚类特征是否存在
    missing_cols = [col for col in cluster_cols if col not in df_all.columns]
    if missing_cols:
        raise ValueError(f"聚类特征列缺失：{missing_cols}，请检查数据列名！")

    # 特征选择函数（可插入到聚类函数的聚类特征定义后）
    def select_cluster_features(df, cluster_cols, var_threshold=0.01, corr_threshold=0.8):
        """特征选择：删除低方差+高相关特征"""
        df_cluster = df[cluster_cols].copy().fillna(df[cluster_cols].median())
        # 1. 删除低方差特征
        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=var_threshold)
        var_selector.fit(df_cluster)
        high_var_cols = df_cluster.columns[var_selector.get_support()].tolist()
        df_high_var = df_cluster[high_var_cols]
        # 2. 删除高相关特征
        corr_matrix = df_high_var.corr()
        high_corr_cols = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    high_corr_cols.add(corr_matrix.columns[i])
        final_cols = [col for col in high_var_cols if col not in high_corr_cols]
        print(f"原始特征：{cluster_cols} → 筛选后特征：{final_cols}")
        return final_cols

    # 在聚类函数中调用：
    cluster_cols = select_cluster_features(df_all, cluster_cols)

    # 2. 鲁棒预处理（替换原简单预处理）
    X_cluster, df_cluster, scaler = preprocess_for_clustering(df_all, cluster_cols)
    if len(df_cluster) == 0:
        raise ValueError("聚类数据为空！请检查数据清洗逻辑")

    # 3. 自动选择最优K值（替换原固定K=3）
    optimal_k = find_optimal_k(X_cluster)

    # 4. 训练K-Means（提升稳定性）
    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=20,  # 增加初始化次数
        max_iter=500
    )
    df_cluster["cluster"] = kmeans.fit_predict(X_cluster)

    # 5. 合并聚类结果到原始数据
    df_all = df_all.merge(df_cluster["cluster"], left_index=True, right_index=True, how="left")

    # 6. 聚类结果解释（新增）
    cluster_stats, cluster_labels = interpret_clusters(df_cluster, cluster_cols)

    # 7. 基础统计结果
    print("\n===== K-Means 聚类结果（每类均值）=====")
    print(cluster_stats)

    # 7. TSNE降维可视化（原有代码保留）
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

    # 新增：饼图可视化（修复cmap问题）
    cluster_counts = df_cluster["cluster"].value_counts()
    labels = [f"聚类 {i}" for i in cluster_counts.index]
    # 从viridis色板中提取对应数量的颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))

    plt.figure(figsize=(8, 6))
    # 替换cmap为colors，移除无效参数
    plt.pie(cluster_counts, labels=labels, autopct="%1.1f%%",
            startangle=90, colors=colors)
    plt.title(f"各聚类样本占比（K={optimal_k}）")
    plt.axis("equal")  # 保证饼图是正圆形
    plt.show()

    # 8. 可视化升级
    # 8.1 TSNE降维可视化
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

    # 8.2 聚类占比饼图
    plt.figure(figsize=(8, 8))
    cluster_counts = df_cluster["cluster"].value_counts()
    labels = [cluster_labels.get(cid, f"聚类{cid}") for cid in cluster_counts.index]
    plt.pie(cluster_counts, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("水质聚类类别占比")
    plt.show()

    # 8.3 特征雷达图
    plot_cluster_radar(cluster_stats, cluster_cols, cluster_labels)

    return df_all, kmeans

# 执行优化后的聚类
df_all_with_cluster, kmeans_model = water_quality_clustering(df_all)

# 查看每个聚类的样本数量
print("\n===== 各聚类样本数量 =====")
print(df_all_with_cluster["cluster"].value_counts())