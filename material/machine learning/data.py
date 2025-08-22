###########数据预处理#########
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('alloys.csv')

# 检查缺失值
print(df.isnull().sum())

# 填补缺失值（例如，用均值填补）
df.fillna(df.mean(), inplace=True)

# 特征与目标变量
X = df[['成分1', '成分2', '成分3', '成分4', '成分5', '润湿性', '硬度']]
y = df['抗拉强度']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
