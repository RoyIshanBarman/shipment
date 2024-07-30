import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


sales_data = pd.read_csv('shop_sale.csv')
reviews_data = pd.read_csv('shop_reviews.csv')



sales_data['Shop_ID'] = sales_data['Shop_ID'].str.replace(' ', '_')
reviews_data['Shop_ID'] = reviews_data['Shop_ID'].str.replace(' ', '_')

merged_data = pd.merge(sales_data, reviews_data, on='Shop_ID')




if merged_data.empty:
    raise ValueError("The merged DataFrame is empty. Please check the input CSV files for consistency.")


x = pd.get_dummies(merged_data.drop(['Shop_ID', 'Month', 'Total_Sales_Amount', 'Review Text', 'Review ID'], axis=1))
y = merged_data['Total_Sales_Amount']


if x.empty or y.empty:
    raise ValueError("Feature set or target variable is empty. Please check the data processing steps.")


x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)


x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


class RankingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RankingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


input_size = x_train_tensor.shape[1]
hidden_size = 64
model = RankingModel(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor.unsqueeze(1))
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')


model.eval()
with torch.no_grad():
    test_outputs = model(x_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor.unsqueeze(1))
print(f'Test Loss: {test_loss.item()}')


torch.save(model.state_dict(), 'ranking_model.pth')


model = RankingModel(input_size, hidden_size)
model.load_state_dict(torch.load('ranking_model.pth'))


features = pd.get_dummies(merged_data.drop(['Shop_ID', 'Month', 'Total_Sales_Amount', 'Review Text', 'Review ID'], axis=1)).values
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
