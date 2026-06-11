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
x = df[["age","gender","school_type","parent_education",
        "study_hours","attendance_percentage","internet_access",
        "travel_time","extra_activities","study_method"]].values

print("数据前5行：")
print(df.head())
# 目标变量：final_grade（分类任务：预测成绩等级A-F）
y = df["final_grade"]

# 特征变量：排除目标列，保留所有输入特征
X = df.drop("final_grade", axis=1)

# 划分训练集/测试集（8:2）
X_train, X_test, y_train, y_test =(train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y))

#二分类完本
binary_features = ["internet_access", "extra_activities"]
# 多分类无序特征（无等级关系）
nominal_features = ["gender", "school_type", "study_method"]
# 多分类有序特征（有等级关系）
ordinal_features = [
    "parent_education",  # 等级：no formal < high school < diploma < graduate < post graduate < phd
    "travel_time"        # 等级：<15 min < 15-30 min < 30-60 min < >60 min
]
# 2. 定义各类型特征的转换逻辑
preprocessor = ColumnTransformer(
    transformers=[
        # 二分类特征：标签编码（0/1）
        ("binary", OrdinalEncoder(), binary_features),
        # 多分类无序特征：独热编码（创建虚拟变量）
        ("nominal", OneHotEncoder(sparse_output=False, drop="first"), nominal_features),
        # 多分类有序特征：有序编码（按等级分配数值）
        ("ordinal", OrdinalEncoder(categories=[
            # parent_education 的等级顺序
            ["no formal", "high school", "diploma", "graduate", "post graduate", "phd"],
            # travel_time 的等级顺序
            ["<15 min", "15-30 min", "30-60 min", ">60 min"]
        ]), ordinal_features)
    ],
    # 保留数值特征（如 age, study_hours, attendance_percentage）不转换
    remainder="passthrough"
)
#logistics回归模型
""""
model = Pipeline(steps =[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=100,random_state=42)),
        ])"""
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
    "study_hours": [6.1],
    "attendance_percentage": [90.5],
    "internet_access": ["yes"],
    "travel_time": ["15-30 min"],
    "extra_activities": ["yes"],
        #17
    "study_method": ["notes"]
})

# 模型预测成绩！
result = model.predict(new_student)
print("预测成绩等级：", result[0])


#高级男女对比
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. 统计男女数量 & 平均成绩等级
# ==============================
print("===== 男女学习行为对比 =====")
gender_group = df.groupby("gender")

# 1.1 人数对比
print("\n【男女人数】")
print(gender_group.size())

# 1.2 成绩等级分布对比
print("\n【男女成绩等级分布】")
print(gender_group["final_grade"].value_counts())

# ==============================
# 2. 关键指标对比（出勤、学习时长、成绩）
# ==============================
print("\n===== 男女学习表现对比 =====")
compare_cols = ["study_hours", "attendance_percentage"]
print(gender_group[compare_cols].mean())

# ==============================
# 3. 画图：男女成绩分布对比
# ==============================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.figure(figsize=(10, 5))

sns.countplot(x="final_grade", hue="gender", data=df)
plt.title("男女学生成绩等级分布对比")
plt.xlabel("成绩等级")
plt.ylabel("人数")
plt.show()
"""
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x, y)
y_pred = reg.predict(x)
y_class = (y_pred >= 0.5).astype(int)
print(reg.coef_)
print(reg.intercept_)
print("前五行")

print("前10个预测值：", y_pred[:10])
print("前10个分类结果：", y_class[:10])
"""