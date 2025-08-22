import tushare as ts
import matplotlib.pyplot as plt

# 初始化Tushare
ts.set_token('226d8041c6f20e093719df03ab6509f6d6dde36c576e3f01174c735e')  # 需要注册Tushare账号并获取token
pro = ts.pro_api()

# 获取指数历史数据
df = pro.index_daily(ts_code='000001.SH', start_date='19901219', end_date='20231019')

# 数据处理，转换日期格式
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values('trade_date')

# 绘制趋势图
plt.figure(figsize=(14, 7))
plt.plot(df['trade_date'], df['close'], label='上证指数')
plt.title('上证指数30年趋势图')
plt.xlabel('年份')
plt.ylabel('指数')
plt.grid(True)
plt.legend()
plt.show()
