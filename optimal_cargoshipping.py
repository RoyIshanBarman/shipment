import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import productorder

class RankingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RankingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def simulate_cargo_shipping(product_name, order_date):
    delivery_time_mean = 48
    delivery_time_std = 12
    delivery_time = max(1, int(np.random.normal(delivery_time_mean, delivery_time_std)))
    delivery_dates = [order_date + timedelta(hours=i) for i in range(delivery_time)]

    data = []
    delivered = False
    for timestamp in delivery_dates:
        if not delivered:
            data.append({
                'Timestamp': timestamp,
                'Status': 'In Transit' if timestamp < delivery_dates[-1] else 'Delivered'
            })
            if timestamp >= delivery_dates[-1]:
                delivered = True
        else:
            data.append({
                'Timestamp': timestamp,
                'Status': 'Delivered'
            })
            break

    tracking_df = pd.DataFrame(data)
    return tracking_df

def parse_order_date(order_date):
    if pd.isna(order_date):
        return None
    if isinstance(order_date, str):
        try:
            return pd.to_datetime(order_date)
        except:
            return None
    return pd.to_datetime(order_date, errors='coerce')

def rank_stores(sales_data_file, reviews_data_file):
    sales_data = pd.read_csv(sales_data_file)
    reviews_data = pd.read_csv(reviews_data_file)

    sales_data['Shop_ID'] = sales_data['Shop_ID'].str.replace(' ', '_')
    reviews_data['Shop_ID'] = reviews_data['Shop_ID'].str.replace(' ', '_')

    merged_data = pd.merge(sales_data, reviews_data, on='Shop_ID')

    if merged_data.empty:
        raise ValueError("The merged DataFrame is empty. Please check the input CSV files for consistency.")

    x = pd.get_dummies(merged_data.drop(['Shop_ID', 'Month', 'Total_Sales_Amount', 'Review Text', 'Review ID'], axis=1))
    y = merged_data['Total_Sales_Amount']

    input_size = x.shape[1]
    hidden_size = 64

    model = RankingModel(input_size, hidden_size)
    model.load_state_dict(torch.load('ranking_model.pth'))

    features = x.values
    features_tensor = torch.tensor(features, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_sales = model(features_tensor)

    predicted_sales = predicted_sales.squeeze().tolist()
    stores = merged_data['Shop_ID'].tolist()
    store_sales_dict = dict(zip(stores, predicted_sales))

    ranked_stores = sorted(store_sales_dict.items(), key=lambda x: x[1], reverse=True)

    return ranked_stores

def rename_shop(shop_id):
    shop_mapping = {'shop1': 'Shop_B', 'shop2': 'Shop_C', 'shop3': 'Shop_A'}
    return shop_mapping.get(shop_id, shop_id)

def cargo_tracking_main():
    product_name_input = input("Enter the product name: ")
    productorder.take_orders(product_name_input)
    orders_df = productorder.load_orders()
    print("Debug: Loaded orders:")
    print(orders_df)

    orders_df['Shop'] = orders_df['Shop'].apply(rename_shop)

    ranked_stores = rank_stores('shop_sale.csv', 'shop_reviews.csv')

    ranked_stores = [(rename_shop(store), score) for store, score in ranked_stores]
    print("Ranking of stores:")
    for rank, (store, _) in enumerate(ranked_stores, start=1):
        print(f"{rank}. {store}")

    product_orders = orders_df[orders_df['Product Name'] == product_name_input]

    if product_orders.empty:
        print(f"No orders placed for {product_name_input}.")
        return

    for store, _ in ranked_stores:
        store_orders = product_orders[product_orders['Shop'] == store]
        for index, order in store_orders.iterrows():
            shop = order['Shop']
            order_date = parse_order_date(order['Order Date'])
            order_quantity = order.get('Order Quantity', 'N/A')
            print(f"Debug: Processing order - Shop: {shop}, Order Date: {order_date}, Order Quantity: {order_quantity}")
            
            if pd.notna(order_date):
                print(f"Reorder needed for {product_name_input} in {shop} on {order_date} for {order_quantity} units")
                tracking_df = simulate_cargo_shipping(product_name_input, order_date)
                print(tracking_df)
            else:
                print(f"Invalid or missing order date for {product_name_input} in {shop}")

if __name__ == "__main__":
    cargo_tracking_main()
