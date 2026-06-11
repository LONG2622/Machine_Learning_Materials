import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

# 1. 读取并准备数据（你熟悉的部分）
df = pd.read_csv('student_grade.csv')

# 特征X 和 标签y
X = df.drop('final_grade', axis=1)
y = df['final_grade']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征预处理（把文字转数字）
binary_features = ["internet_access", "extra_activities"]
nominal_features = ["gender", "school_type", "study_method"]
ordinal_features = ["parent_education", "travel_time"]
numeric_features = ["age", "study_hours", "attendance_percentage"]

preprocessor = ColumnTransformer([
    ("binary", OrdinalEncoder(), binary_features),
    ("nominal", OneHotEncoder(sparse_output=False, drop="first"), nominal_features),
    ("ordinal", OrdinalEncoder(categories=[
        ["no formal", "high school", "diploma", "graduate", "post graduate", "phd"],
        ["<15 min", "15-30 min", "30-60 min", ">60 min"]
    ]), ordinal_features),
    ("num", StandardScaler(), numeric_features)
])

# 先把数据处理成神经网络能吃的数字
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 把成绩 A-F 也转成数字
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

#深度学习神经网络
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. 建造大脑 (Sequential = 一层一层的神经网络)
model = Sequential([
    # 第一层：64个神经元 (脑细胞)，接收输入特征
    Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    Dropout(0.3),  # 防止走神

    # 第二层：32个神经元 (深层思考)
    Dense(32, activation='relu'),

    # 输出层：预测6个等级 (A-F)
    Dense(6, activation='softmax')
])

# 2. 编译大脑 (设定学习规则)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 打印模型结构（看看你的AI长什么样）
print("\n==== 你的深度学习AI 结构 ====")
model.summary()

# 3. 开始训练 (让AI看书学习)
print("\n开始训练神经网络...")
model.fit(
    X_train_processed, y_train_encoded,
    epochs=50,  # 学习50遍
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# 4. 考试评分
print("\n==== 深度学习模型成绩 ====")
test_loss, test_acc = model.evaluate(X_test_processed, y_test_encoded)
print(f"深度学习准确率: {test_acc:.2f}")