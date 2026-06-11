import os
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from sklearn.pipeline import Pipeline

# 改成你电脑真正的路径
train_root = r"D:\Conda\Machine-Leaning\PythonProject\训练集"
test_root = r"D:\Conda\Machine-Leaning\PythonProject\测试集"

#读取训练集测试集的文件
def load_all_data(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.endswith('.xlsx'):
            file_path = os.path.join(folder, f)
            df = pd.read_excel(file_path, engine='openpyxl')
            dfs.append(df)
    return pd.concat(dfs,ignore_index=True)

all_train = load_all_data(train_root)
all_test = load_all_data(test_root)

all_train = all_train.select_dtypes(include=['int64', 'float64'])
all_test = all_test.select_dtypes(include=['int64', 'float64'])

#确定特征X 和 标签y

label_col = "Temp_C"
"""
编码
SBF5Status,Hmax_m,Abs_Tilt,Tsig,TDS_ppt,Turb_NTU,SBF5Temp,RIntensity,Ref_V,longitude,Tilt_X,Havg_m,Std_Tilt,NH3_ODS,Depth_m,
PO4_ODS,Pressure,Tilt_Y,HEADING,SpCond_mS,Temp_Enc,NO3_Conc,NO3_ODE,DO_percent,DO_ppm，NH3_ODE,HIntensity，Strength,Tavg，
Supply_V,H10_m，PE_ppb,Chl_ppb，openStatus,MeanDirection，RH_Enc,Chl_RFU，pH,WDmin，WDmax,Ping_count，WSmin,Batt_Volt，
WDavg,RH,Direction_current，WSmax,Tmax,East_speed，Max_Tilt,pH_mV，PE_RFU,AirTC,NH3_Conc,
Rain_mm,Temp_C，SP_std,NO2_ODE，Hduration,PO4_Conc,Rduration,Hsig_m,Version,latitude,SBF5V,PO4_ODE,
Heading_deg,Cond_mS,Hamount,NO3_ODS,NO2_Conc,North_speed,Abs_speed,NO2_ODS,T10,Sal,WSavg,ITemp
"""
#提取特征和标签
X_train = all_train.drop(label_col , axis=1)
y_train = all_train[label_col]
X_test = all_test.drop(label_col , axis=1)
y_test = all_test[label_col]

#标签空值填充

y_train = y_train.fillna(y_train.median()) # mean
y_test = y_test.fillna(y_test.median())

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

#自动填充所有空值
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

y_train = y_train.fillna(y_train.median())
y_test = y_test.fillna(y_test.median())

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)

#数据预处理
#线性回归模型
model = Pipeline([("scaler", StandardScaler()),
                  ("model", LinearRegression())])
model.fit(X_train, y_train)
pred_lr = model.predict(X_test)

print(f"R2: {r2_score(y_test, pred_lr):.4f}")

#决策树模型
dt_pipeline = Pipeline([
    ("classifier", DecisionTreeRegressor(random_state=42,max_depth=10))])
dt_pipeline.fit(X_train, y_train)
pred_dt = dt_pipeline.predict(X_test)
y_pred_dt = dt_pipeline.predict(X_test)
dt_r2 = r2_score(y_test,pred_dt)
print(f"R2: {r2_score(y_test,pred_dt):.4f}")

score_lr = r2_score(y_test, pred_lr)
score_dt = r2_score(y_test, pred_dt)
#最终结果和比较
print(f"线性回归  R² 分数: {score_lr:.4f}")
print(f"决策树回归 R² 分数: {score_dt:.4f}")

best_name = "线性回归" if score_lr > score_dt else "决策树回归"
best_score = max(score_lr, score_dt)
print(f"\n 最优模型：{best_name}")
print(f"最优预测分数：{best_score:.4f}")
print("\n 全部运行成功！")
"""
#数据可视化处理
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['Microsoft Yi']
plt.rcParams['axes.unicode_minus'] = False
#真实值与预测值对比图
ply.figure(figsize=(10,5))
plt.scatter(y_test, pred_lr, color='blue',alpha=0.5,label='线性回归预测')
plt.scatter(y_test,pred_dt,alpha = 0.5, color='red',label='决策树预测')
plt.xlabel('real temp')
plt.ylabel('predict temp')
pli.title(labei_col,'真实值与预测值对比')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(label_col,dpi=300,'真实值与预测值对比.png')
plt.show()
"""

# 数据可视化（不报错版）
# ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 5))
plt.scatter(y_test, pred_lr, color='blue', alpha=0.5, label='线性回归预测')
plt.scatter(y_test, pred_dt, alpha=0.5, color='red', label='决策树预测')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('真实温度')
plt.ylabel('预测温度')
plt.title(f'{label_col} 真实值 vs 预测值 对比')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f'{label_col}_真实值_预测值.png', dpi=300, bbox_inches='tight')
plt.show()