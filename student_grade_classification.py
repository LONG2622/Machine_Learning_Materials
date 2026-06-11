import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("student_grade.csv")

print("数据前5行：")
print(df.head())

print("\n数据基本信息：")
print(df.info())

print("\n是否存在缺失值：")
print(df.isnull().sum())

print("\nfinal_grade 各类别数量：")
print(df["final_grade"].value_counts())

# =========================
# 2. 特征与标签划分
# =========================
X = df.drop(columns=["final_grade"])
y = df["final_grade"]

# =========================
# 3. 标签编码
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\n类别编码对应关系：")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {i}")

# =========================
# 4. 类别特征独热编码
# =========================
X = pd.get_dummies(X, drop_first=True)

print("\n编码后的特征维度：", X.shape)
print("编码后的特征列：")
print(X.columns.tolist())

# =========================
# 5. 划分训练集和测试集
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\n训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)

# =========================
# 6. 特征标准化（逻辑回归需要）
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 7. 逻辑回归分类
# =========================
print("\n=========================")
print("逻辑回归分类结果")
print("=========================")

lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

lr_acc = accuracy_score(y_test, y_pred_lr)
print("逻辑回归准确率：", lr_acc)

print("\n逻辑回归分类报告：")
print(classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(7, 6))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# =========================
# 8. 决策树分类
# =========================
print("\n=========================")
print("决策树分类结果")
print("=========================")

dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

dt_acc = accuracy_score(y_test, y_pred_dt)
print("决策树准确率：", dt_acc)

print("\n决策树分类报告：")
print(classification_report(y_test, y_pred_dt, target_names=label_encoder.classes_))

cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(7, 6))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Greens",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# =========================
# 9. 模型结果对比
# =========================
print("\n=========================")
print("模型准确率对比")
print("=========================")
print(f"逻辑回归准确率: {lr_acc:.4f}")
print(f"决策树准确率: {dt_acc:.4f}")