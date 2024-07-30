import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from datetime import datetime, timedelta

# Load and preprocess data
sales_data = pd.read_csv('shop_sale.csv')
reviews_data = pd.read_csv('shop_reviews.csv')

sales_data['Shop_ID'] = sales_data['Shop_ID'].str.replace(' ', '_')
reviews_data['Shop_ID'] = reviews_data['Shop_ID'].str.replace(' ', '_')

merged_data = pd.merge(sales_data, reviews_data, on='Shop_ID')

if merged_data.empty:
    raise ValueError("The merged DataFrame is empty. Please check the input CSV files for consistency.")

X = pd.get_dummies(merged_data.drop(['Shop_ID', 'Month', 'Total_Sales_Amount', 'Review Text', 'Review ID'], axis=1))
y = merged_data['Total_Sales_Amount']

if X.empty or y.empty:
    raise ValueError("Feature set or target variable is empty. Please check the data processing steps.")

# Split data and scale features
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the model
class RankingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RankingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
input_size = X_train_tensor.shape[1]
hidden_size = 64
model = RankingModel(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 200
best_val_loss = float('inf')
patience = 20
no_improve = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_ranking_model.pth')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping")
            break

# Load best model and evaluate
model.load_state_dict(torch.load('best_ranking_model.pth'))
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
print(f'Test Loss: {test_loss.item():.4f}')

# Predict sales and rank stores
features = scaler.transform(pd.get_dummies(merged_data.drop(['Shop_ID', 'Month', 'Total_Sales_Amount', 'Review Text', 'Review ID'], axis=1)).values)
features_tensor = torch.tensor(features, dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_sales = model(features_tensor)

predicted_sales = predicted_sales.squeeze().tolist()
stores = merged_data['Shop_ID'].tolist()
store_sales_dict = dict(zip(stores, predicted_sales))

ranked_stores = sorted(store_sales_dict.items(), key=lambda x: x[1], reverse=True)

print("Ranking of stores:")
for rank, (store, _) in enumerate(ranked_stores, start=1):
    print(f"{rank}. {store}")

# Order processing functions
def process_quick_order(order_date, quantity):
    delivery_time = timedelta(hours=random.uniform(6, 12))
    delivery_date = order_date + delivery_time
    
    tracking = []
    current_time = order_date
    while current_time <= delivery_date:
        status = "In Transit" if current_time < delivery_date else "Delivered"
        tracking.append((current_time, status, "Quick Order"))
        current_time += timedelta(minutes=30)  # Update every 30 minutes
    
    return tracking

def process_normal_order(order_date, quantity):
    delivery_time = timedelta(hours=random.uniform(24, 48))
    delivery_date = order_date + delivery_time
    
    tracking = []
    current_time = order_date
    while current_time <= delivery_date:
        status = "In Transit" if current_time < delivery_date else "Delivered"
        tracking.append((current_time, status, "Normal Order"))
        current_time += timedelta(hours=1)  # Update every hour
    
    return tracking

# Example usage of order processing
order_date = datetime.now()
quick_order_tracking = process_quick_order(order_date, 5)
normal_order_tracking = process_normal_order(order_date, 10)

print("\nQuick Order Tracking:")
for timestamp, status, order_type in quick_order_tracking:
    print(f"{timestamp}: {status} - {order_type}")

print("\nNormal Order Tracking:")
for timestamp, status, order_type in normal_order_tracking:
    print(f"{timestamp}: {status} - {order_type}")