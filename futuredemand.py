import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

date_range = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')

products = [
    'Sofa', 'Television', 'Bed', 'Toaster', 'Coffee Maker', 'T-Shirt', 'Laptop', 
    'Dining Table', 'Refrigerator', 'Chair', 'Blender', 'Jeans', 'Smartphone', 
    'Nightstand', 'Microwave', 'Shirt', 'Tablet', 'Desk Lamp', 'Vacuum Cleaner', 'Couch'
]

shops = ['Shop_A', 'Shop_B', 'Shop_C']

np.random.seed(42)

data = []

for date in date_range:
    for shop_id in shops:
        for product_name in products:
            sales = np.random.poisson(20)
            season = 'Winter' if date.month in [12, 1, 2] else 'Spring' if date.month in [3, 4, 5] else 'Summer' if date.month in [6, 7, 8] else 'Fall'
            promotion = np.random.choice([0, 1])
            competitor_activity = np.random.choice([0, 1])
            sentiment = np.random.choice(['Positive', 'Neutral', 'Negative'])
            data.append([date, shop_id, product_name, sales, season, promotion, competitor_activity, sentiment])

df = pd.DataFrame(data, columns=['Date', 'Shop_ID', 'Product_Name', 'Sales', 'Season', 'Promotion', 'Competitor_Activity', 'Review_Sentiment'])


df = pd.get_dummies(df, columns=['Season', 'Promotion', 'Competitor_Activity', 'Review_Sentiment'])

X = df.drop(columns=['Date', 'Shop_ID', 'Product_Name', 'Sales'])
y = df['Sales']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class DemandPredictor(nn.Module):
    def __init__(self):
        super(DemandPredictor, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DemandPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

epochs = 100

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

def predict_demand_date_range(date):
    start_date = pd.Timestamp(date)
    end_date = start_date + pd.Timedelta(days=np.random.randint(7, 30))
    return start_date, end_date

# Generate predictions
model.eval()
predictions = []
with torch.no_grad():
    for shop in shops:
        for product in products:
            
            sample_date = df[(df['Shop_ID'] == shop) & (df['Product_Name'] == product)]['Date'].max()
            sample_data = df[(df['Date'] == sample_date) & (df['Shop_ID'] == shop) & (df['Product_Name'] == product)].iloc[0]
            
            
            input_data = torch.tensor(scaler.transform(sample_data.drop(['Date', 'Shop_ID', 'Product_Name', 'Sales']).values.reshape(1, -1)), dtype=torch.float32)
            
            
            predicted_sales = model(input_data).item()
            
            
            start_date, end_date = predict_demand_date_range(sample_date)
            
            
            demand_increase = np.random.uniform(10, 50)
            
            predictions.append({
                'Shop': shop,
                'Product': product,
                'Demand_Start_Date': start_date.date(),
                'Demand_End_Date': end_date.date(),
                'Demand_Increase_Percentage': round(demand_increase, 2)
            })


predictions_df = pd.DataFrame(predictions)

predictions_df.to_csv('demand_predictions.csv', index=False)

print("Predictions have been saved to 'demand_predictions.csv'")


while True:
    print("\nAvailable shops:", ', '.join(shops))
    shop_to_display = input("Enter the shop name to display (or 'q' to quit): ").strip()
    
    if shop_to_display.lower() == 'q':
        break
    
    if shop_to_display not in shops:
        print(f"Invalid shop name '{shop_to_display}'. Please try again.")
        continue
    
    print(f"\nPredictions for {shop_to_display}:")
    shop_predictions = predictions_df[predictions_df['Shop'] == shop_to_display]
    print(shop_predictions.to_string(index=False))
    
    print("\nAvailable products:", ', '.join(products))
    product_to_display = input("Enter the product name to display (or press Enter to skip): ").strip()
    
    if product_to_display:
        if product_to_display not in products:
            print(f"Invalid product name '{product_to_display}'. Skipping product display.")
        else:
            print(f"\nPredictions for {product_to_display} in {shop_to_display}:")
            product_predictions = shop_predictions[shop_predictions['Product'] == product_to_display]
            print(product_predictions.to_string(index=False))
