import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import productorder
import hashlib
import json
import base64
from cryptography.fernet import Fernet

# Blockchain encryption setup
key = Fernet.generate_key()
cipher_suite = Fernet(key)

class RankingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RankingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def simulate_cargo_shipping(product_name, order_date, is_quick_order=False):
    if is_quick_order:
        delivery_time_mean = 48  # Hours
        delivery_time_std = 12
    else:
        delivery_time_mean = 120  # Hours
        delivery_time_std = 24
    
    delivery_time = max(1, int(np.random.normal(delivery_time_mean, delivery_time_std)))
    delivery_dates = [order_date + timedelta(hours=i) for i in range(delivery_time)]

    data = []
    delivered = False
    for timestamp in delivery_dates:
        status = 'Delivered' if timestamp >= delivery_dates[-1] else 'In Transit'
        data.append({
            'Timestamp': timestamp,
            'Status': status,
            'Order Type': 'Quick Order' if is_quick_order else 'Normal Order'
        })
        if status == 'Delivered':
            delivered = True

    tracking_df = pd.DataFrame(data)
    return tracking_df

def load_sales_data():
    try:
        shop_1 = pd.read_csv('shop_1_combined.csv')
        shop_2 = pd.read_csv('shop_2.csv')
        shop_3 = pd.read_csv('shop_3.csv')
        
        sales_data = pd.concat([shop_1, shop_2, shop_3], ignore_index=True)
        return sales_data
    except Exception as e:
        print(f"Error loading sales data: {e}")
        return None

def calculate_excess_inventory(sales_data):
    product_summary = sales_data.groupby('Product Name').agg({
        'Amount Sold': 'sum',
        'Visible Stock': 'last',
        'Inventory': 'last'
    }).reset_index()

    product_summary['Excess Inventory'] = product_summary['Inventory'] - product_summary['Amount Sold']
    product_summary['Excess Inventory'] = product_summary['Excess Inventory'].clip(lower=0)

    excess_inventory = dict(zip(product_summary['Product Name'], product_summary['Excess Inventory']))
    
    return excess_inventory

def process_quick_order(product_name, shop, order_date, order_quantity, excess_inventory):
    print(f"Processing order for {product_name} in {shop}")
    
    max_quick_order = int(order_quantity * 0.3)
    
    print(f"You can order up to {max_quick_order} units as a quick order (30% of total).")
    quick_order_quantity = input(f"Enter quick order quantity for {shop} (max {max_quick_order}): ")
    
    try:
        quick_order_quantity = int(quick_order_quantity)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        quick_order_quantity = 0

    if quick_order_quantity > max_quick_order:
        print(f"Quick order quantity exceeds 30% limit. Adjusting to {max_quick_order} units.")
        quick_order_quantity = max_quick_order
    
    normal_order_quantity = order_quantity - quick_order_quantity
    
    if quick_order_quantity > 0 and product_name in excess_inventory and excess_inventory[product_name] >= quick_order_quantity:
        print(f"Transferring {quick_order_quantity} units from excess inventory as quick order.")
        excess_inventory[product_name] -= quick_order_quantity
        quick_tracking_df = pd.DataFrame({
            'Timestamp': [order_date],
            'Status': ['Transferred'],
            'Order Type': ['Quick Order'],
            'From': ['Excess Inventory'],
            'To': [shop],
            'Quantity': [quick_order_quantity]
        })
    else:
        print(f"No sufficient excess inventory found or invalid quick order. Processing as new quick order.")
        quick_tracking_df = simulate_cargo_shipping(product_name, order_date, is_quick_order=True)
    
    print("Quick Order Tracking:")
    print(quick_tracking_df)
    
    normal_tracking_df = simulate_cargo_shipping(product_name, order_date, is_quick_order=False)
    print("Normal Order Tracking:")
    print(normal_tracking_df)
    
    tracking_df = pd.concat([quick_tracking_df, normal_tracking_df], ignore_index=True)
    
    return tracking_df

def rename_shop(shop_id):
    shop_mapping = {'shop1': 'Shop_A', 'shop2': 'Shop_B', 'shop3': 'Shop_C'}
    return shop_mapping.get(shop_id, shop_id)

def parse_order_date(order_date):
    if pd.isna(order_date):
        return None
    if isinstance(order_date, str):
        try:
            parsed_date = pd.to_datetime(order_date)
            return parsed_date
        except Exception as e:
            print(f"Failed to parse string date: {e}")
            return None
    parsed_date = pd.to_datetime(order_date, errors='coerce')
    return parsed_date

def rank_stores(sales_data_file, reviews_data_file):
    try:
        sales_data = pd.read_csv(sales_data_file)
        reviews_data = pd.read_csv(reviews_data_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return []

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
    try:
        model.load_state_dict(torch.load('ranking_model.pth'))
    except Exception as e:
        print(f"Error loading model state: {e}")
        return []

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

def save_demand_data(demand_data, filename="demand_data.csv"):
    demand_data.to_csv(filename, index=False)
    print(f"Demand data saved to {filename}")

def parse_order_quantity(order_quantity):
    try:
        if isinstance(order_quantity, str) and order_quantity.startswith('['):
            quantity_list = eval(order_quantity)
            return sum(quantity_list) if isinstance(quantity_list, list) else quantity_list
        return float(order_quantity)  
    except Exception as e:
        print(f"Error parsing order quantity: {e}")
        return 0

### Blockchain Invoice Functions ###
def generate_invoice_hash(order_data):
    invoice_string = f"{order_data['Product Name']}{order_data['Shop']}{order_data['Order Date']}{order_data['Order Quantity']}"
    invoice_hash = hashlib.sha256(invoice_string.encode()).hexdigest()
    return invoice_hash

def create_blockchain_invoice(product_name, shop, order_date, order_quantity, tracking_df):
    invoice = {
        'Product Name': product_name,
        'Shop': shop,
        'Order Date': order_date,
        'Order Quantity': order_quantity,
        'Tracking Data': tracking_df.to_dict('records')
    }
    
    invoice_json = json.dumps(invoice, default=str)
    encrypted_invoice = cipher_suite.encrypt(invoice_json.encode())
    invoice_hash = generate_invoice_hash(invoice)
    invoice['Invoice Hash'] = invoice_hash

    print(f"Blockchain Invoice for {product_name} at {shop} on {order_date} generated with hash: {invoice_hash}")
    return invoice

def cargo_tracking_main():
    sales_data = load_sales_data()
    if sales_data is None:
        print("Failed to load sales data. Exiting.")
        return

    excess_inventory = calculate_excess_inventory(sales_data)

    product_name_input = input("Enter the product name: ")
    productorder.take_orders(product_name_input)
    order_date = datetime.now()

    order_quantity = parse_order_quantity(input("Enter order quantity: "))
    if order_quantity <= 0:
        print("Invalid order quantity. Exiting.")
        return

    shop_id = input("Enter shop ID (shop1, shop2, shop3): ").strip().lower()
    shop = rename_shop(shop_id)
    
    tracking_df = process_quick_order(product_name_input, shop, order_date, order_quantity, excess_inventory)

    invoice = create_blockchain_invoice(product_name_input, shop, order_date, order_quantity, tracking_df)
    
    print("Invoice created:")
    print(json.dumps(invoice, indent=4, default=str))

if __name__ == "__main__":
    cargo_tracking_main()
