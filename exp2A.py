#实验二   基于学生学习行为的成绩等级分类
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('student_grade.csv')
x = df[["age","gender","school_type","parent_education"]]

print("数据前5行：")
print(df.head())
# 目标变量：final_grade（分类任务：预测成绩等级A-F）
y = df["final_grade"]

# 特征变量：排除目标列，保留所有输入特征
#X = df.drop("final_grade", axis=1)
X = df[["age", "gender", "school_type", "parent_education"]]
# 划分训练集/测试集（8:2）
X_train, X_test, y_train, y_test =(train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y))

#二分类完本
#binary_features = ["internet_access", "extra_activities"]
# 多分类无序特征（无等级关系）
nominal_features = ["gender", "school_type"]#, "study_method"]
# 多分类有序特征（有等级关系）
ordinal_features = [
    "parent_education",  # 等级：no formal < high school < diploma < graduate < post graduate < phd
]
# 2. 定义各类型特征的转换逻辑
preprocessor = ColumnTransformer(
    transformers=[
        # 多分类无序特征：独热编码（创建虚拟变量）
        ("nominal", OneHotEncoder(sparse_output=False, drop="first"), nominal_features),
        # 多分类有序特征：有序编码（按等级分配数值）
        ("ordinal", OrdinalEncoder(categories=[
            # parent_education 的等级顺序
            ["no formal", "high school", "diploma", "graduate", "post graduate", "phd"],
        ]), ordinal_features)
    ],
    # 保留数值特征（如 age, study_hours, attendance_percentage）不转换
    remainder="passthrough"
)
#随机森林
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy:{accuracy:.2f}")

#=======测试=================
new_student = pd.DataFrame({
    "age": [17],
    "gender": ["female"],
    "school_type": ["private"],
    "parent_education": ["graduate"],
})
# 模型预测成绩！
result = model.predict(new_student)
print("预测成绩等级：", result[0])

#混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 设置中文显示（防止中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 画图
plt.figure(figsize=(8, 7))
sns.heatmap(
    cm,
    annot=True,        # 显示数字
    fmt="d",           # 整数格式
    cmap="Blues",      # 蓝色风格
    xticklabels=sorted(df["final_grade"].unique()),
    yticklabels=sorted(df["final_grade"].unique())
)
plt.title("方案A 混淆矩阵（仅背景 准确率0.26）", fontsize=14)
plt.xlabel("模型预测成绩", fontsize=12)
plt.ylabel("真实成绩", fontsize=12)
plt.tight_layout()
plt.show()

# ==============================
# 方案 A 分类报告
# ==============================
from sklearn.metrics import classification_report

print("\n===== 方案 A 分类报告（仅背景特征）=====")
print(classification_report(
    y_test, y_pred,
    target_names=sorted(df["final_grade"].unique()),
    zero_division=0
))