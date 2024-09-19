import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import productorder
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import json

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.create_block(previous_hash='1')
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def create_block(self, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.now()),
            'transactions': self.current_transactions,
            'previous_hash': previous_hash
        }
        self.current_transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, transaction):
        self.current_transactions.append(transaction)
        return self.last_block['index'] + 1

    @property
    def last_block(self):
        return self.chain[-1]

    def hash(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return self.private_key.sign(
            block_string,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

    def save_chain(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chain, f)

class RankingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RankingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def encrypt_data(data, key):
    f = Fernet(key)
    return f.encrypt(json.dumps(data).encode())

def decrypt_data(encrypted_data, key):
    f = Fernet(key)
    return json.loads(f.decrypt(encrypted_data))

def simulate_cargo_shipping(product_name, order_date, is_quick_order=False, encryption_key=None):
    if is_quick_order:
        delivery_time_mean = 48  # hours
        delivery_time_std = 12
    else:
        delivery_time_mean = 120  # hours (5 days)
        delivery_time_std = 24

    delivery_time = max(1, int(np.random.normal(delivery_time_mean, delivery_time_std)))
    delivery_dates = [order_date + timedelta(hours=i) for i in range(delivery_time)]

    data = []
    for timestamp in delivery_dates:
        status = 'Delivered' if timestamp >= delivery_dates[-1] else 'In Transit'
        data.append({
            'Timestamp': str(timestamp),
            'Status': status,
            'Order Type': 'Quick Order' if is_quick_order else 'Normal Order'
        })

    tracking_df = pd.DataFrame(data)
    
    # Encrypt the shipping process data before adding to blockchain
    encrypted_data = encrypt_data({
        'Product': product_name,
        'Order Date': str(order_date),
        'Delivery Dates': [str(date) for date in delivery_dates],
        'Delivery Status': status,
        'Order Type': 'Quick Order' if is_quick_order else 'Normal Order'
    }, encryption_key)
    
    blockchain.add_transaction(encrypted_data)
    
    return tracking_df

def process_quick_order(product_name, shop, order_date, order_quantity, excess_inventory, blockchain, encryption_key):
    print(f"Processing order for {product_name} in {shop}")
    
    max_quick_order = int(order_quantity * 0.3)
    print(f"You can order up to {max_quick_order} units as a quick order (30% of total).")
    
    quick_order_quantity = int(input(f"Enter quick order quantity for {shop} (max {max_quick_order}): "))
    if quick_order_quantity > max_quick_order:
        print(f"Quick order quantity exceeds 30% limit. Adjusting to {max_quick_order} units.")
        quick_order_quantity = max_quick_order
    
    normal_order_quantity = order_quantity - quick_order_quantity
    
    # Generate and encrypt the invoice before adding to blockchain
    invoice = {
        'Shop': shop,
        'Product': product_name,
        'Order Quantity': order_quantity,
        'Quick Order Quantity': quick_order_quantity,
        'Normal Order Quantity': normal_order_quantity,
        'Order Date': str(order_date),
        'Total Amount': quick_order_quantity * 10  # assuming unit price of 10
    }
    encrypted_invoice = encrypt_data(invoice, encryption_key)
    blockchain.add_transaction(encrypted_invoice)
    print(f"Encrypted invoice added to blockchain")
    
    if product_name in excess_inventory and excess_inventory[product_name] >= quick_order_quantity:
        print(f"Transferring {quick_order_quantity} units from excess inventory as quick order.")
        excess_inventory[product_name] -= quick_order_quantity
        quick_tracking_df = pd.DataFrame({
            'Timestamp': [str(order_date)],
            'Status': ['Transferred'],
            'Order Type': ['Quick Order'],
            'From': ['Excess Inventory'],
            'To': [shop],
            'Quantity': [quick_order_quantity]
        })
    else:
        print(f"No sufficient excess inventory found. Processing as new quick order.")
        quick_tracking_df = simulate_cargo_shipping(product_name, order_date, is_quick_order=True, encryption_key=encryption_key)
    
    print("Quick Order Tracking:")
    print(quick_tracking_df)
    
    normal_tracking_df = simulate_cargo_shipping(product_name, order_date, is_quick_order=False, encryption_key=encryption_key)
    print("Normal Order Tracking:")
    print(normal_tracking_df)
    
    tracking_df = pd.concat([quick_tracking_df, normal_tracking_df], ignore_index=True)
    
    return tracking_df

def cargo_tracking_main():
    blockchain = Blockchain()
    encryption_key = Fernet.generate_key()

    sales_data = productorder.load_sales_data()
    if sales_data is None:
        print("Failed to load sales data. Exiting.")
        return

    excess_inventory = productorder.calculate_excess_inventory(sales_data)

    product_name_input = input("Enter the product name: ")
    productorder.take_orders(product_name_input)
    orders_df = productorder.load_orders()

    orders_df['Shop'] = orders_df['Shop'].apply(productorder.rename_shop)

    ranked_stores = productorder.rank_stores('shop_sale.csv', 'shop_reviews.csv')
    ranked_stores = [(productorder.rename_shop(store), score) for store, score in ranked_stores]
    print("Ranking of stores:")
    for rank, (store, _) in enumerate(ranked_stores, start=1):
        print(f"{rank}. {store}")

    product_orders = orders_df[orders_df['Product Name'] == product_name_input]
    product_orders.loc[:, 'Order Quantity'] = product_orders['Order Quantity'].apply(productorder.parse_order_quantity)
    product_orders = product_orders[pd.notna(product_orders['Order Date']) & (product_orders['Order Quantity'] > 0)]
    
    if product_orders.empty:
        print(f"No valid orders placed for {product_name_input}.")
        return

    quick_order_enabled = input("Do you want to enable quick orders? (yes/no): ").lower() == 'yes'
    aggregated_orders = product_orders.groupby('Shop').agg({
        'Order Date': 'first',
        'Order Quantity': 'sum'
    }).reset_index()

    tracking_data = []

    for _, order in aggregated_orders.iterrows():
        shop = order['Shop']
        order_date = productorder.parse_order_date(order['Order Date'])
        order_quantity = int(order['Order Quantity'])
        
        if pd.notna(order_date):
            if quick_order_enabled:
                tracking_df = process_quick_order(product_name_input, shop, order_date, order_quantity, excess_inventory, blockchain, encryption_key)
            else:
                print(f"Normal order for {product_name_input} in {shop} on {order_date} for {order_quantity} units")
                tracking_df = simulate_cargo_shipping(product_name_input, order_date, is_quick_order=False, encryption_key=encryption_key)

            tracking_data.append(tracking_df)

    if tracking_data:
        tracking_data_df = pd.concat(tracking_data, ignore_index=True)
        print("Full Tracking Data:")
        print(tracking_data_df)

    # Save blockchain to a file for reference
    blockchain.save_chain('supply_chain_blockchain.json')
    print("Blockchain has been saved to 'supply_chain_blockchain.json'.")

    # Save encryption key securely (in a real-world scenario, this would be done more securely)
    with open('encryption_key.txt', 'wb') as key_file:
        key_file.write(encryption_key)
    print("Encryption key has been saved to 'encryption_key.txt'. Keep this file secure!")

if __name__ == '__main__':
    cargo_tracking_main()
