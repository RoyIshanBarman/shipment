import pandas as pd
from datetime import datetime,timedelta

df=pd.read_csv('maininven.csv')
df['Expiry Date']=pd.to_datetime(df['Expiry Date'])

today = datetime.now()
threshold_date = today + timedelta(days=30)


normal_orders = df[df['Expiry Date'] > threshold_date]
quick_orders = df[df['Expiry Date'] <= threshold_date]


print("Normal Orders:")
print(normal_orders)

print("\nQuick Orders:")
print(quick_orders)

normal_orders.to_csv('normal_orders.csv', index=False)
quick_orders.to_csv('quick_orders.csv', index=False)
