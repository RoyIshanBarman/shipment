import pandas as pd
import finalstore1
import finalstore2
import finalstore3
from datetime import datetime, timedelta
import numpy as np
import productorder
import json
import os

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.create_block(previous_hash='1')  # Genesis block
        self.key = self.generate_key()
    
    def generate_key(self):
        password = b"supply_chain_secret"
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

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

    def encrypt_data(self, data):
        f = Fernet(self.key)
        return f.encrypt(json.dumps(data).encode())

    def decrypt_data(self, encrypted_data):
        f = Fernet(self.key)
        return json.loads(f.decrypt(encrypted_data).decode())

    def save_chain(self, filename):
        encrypted_chain = [self.encrypt_data(block) for block in self.chain]
        with open(filename, 'wb') as file:
            for block in encrypted_chain:
                file.write(block + b'\n')

    def load_chain(self, filename):
        self.chain = []
        with open(filename, 'rb') as file:
            for line in file:
                decrypted_block = self.decrypt_data(line.strip())
                self.chain.append(decrypted_block)

# Preorder management
def process_preorder(product_name, shop, order_date, order_quantity, excess_inventory, blockchain):
    print(f"Processing preorder for {product_name} in {shop}")
    
    # Handle preorder logic
    preorder_quantity = int(input(f"Enter preorder quantity for {shop}: "))
    
    # Create an invoice for the preorder
    invoice = {
        'Shop': shop,
        'Product': product_name,
        'Order Quantity': order_quantity,
        'Preorder Quantity': preorder_quantity,
        'Order Date': str(order_date),
        'Total Amount': preorder_quantity * 10
    }
    blockchain.add_transaction(invoice)
    print(f"Preorder invoice added to blockchain: {invoice}")
    
    if product_name in excess_inventory and excess_inventory[product_name] >= preorder_quantity:
        print(f"Transferring {preorder_quantity} units from excess inventory for preorder.")
        excess_inventory[product_name] -= preorder_quantity
        preorder_tracking_df = pd.DataFrame({
            'Timestamp': [order_date],
            'Status': ['Transferred'],
            'Order Type': ['Preorder'],
            'From': ['Excess Inventory'],
            'To': [shop],
            'Quantity': [preorder_quantity]
        })
    else:
        print(f"No sufficient excess inventory found. Handling as new preorder.")
        preorder_tracking_df = simulate_cargo_shipping(product_name, order_date, is_quick_order=False)
    
    print("Preorder Tracking:")
    print(preorder_tracking_df)
    
    return preorder_tracking_df

def demand_forecasting_for_all_shops(product_name):
    shop1_files = ["shop_1_combined.csv"]
    shop2_file = ["shop_2.csv"]
    shop3_file = ["shop_3.csv"]

    reorder_dates = {}
    try:
        reorder_dates['shop1'] = finalstore1.demand_forecasting_main(shop1_files, product_name)
    except Exception as e:
        print(f"Error forecasting demand for Shop 1: {e}")
        reorder_dates['shop1'] = (None, 0)
    
    try:
        reorder_dates['shop2'] = finalstore2.demand_forecasting_main(shop2_file, product_name)
    except Exception as e:
        print(f"Error forecasting demand for Shop 2: {e}")
        reorder_dates['shop2'] = (None, 0)
    
    try:
        reorder_dates['shop3'] = finalstore3.demand_forecasting_main(shop3_file, product_name)
    except Exception as e:
        print(f"Error forecasting demand for Shop 3: {e}")
        reorder_dates['shop3'] = (None, 0)

    return reorder_dates

def take_orders(product_name):
    reorder_dates = demand_forecasting_for_all_shops(product_name)
    orders_df = load_orders()

    # Remove existing orders for this product
    orders_df = orders_df[orders_df['Product Name'] != product_name]

    new_orders = []
    for shop, (reorder_date, order_quantity) in reorder_dates.items():
        if reorder_date:
            new_order = {
                'Product Name': product_name,
                'Shop': shop,
                'Order Date': reorder_date,
                'Order Quantity': order_quantity
            }
            new_orders.append(new_order)
            print(f"Order placed for {product_name} in {shop} on {reorder_date} for {order_quantity} units")
        else:
            print(f"No reorder needed for {product_name} in {shop}")

    if new_orders:
        new_orders_df = pd.DataFrame(new_orders)
        orders_df = pd.concat([orders_df, new_orders_df], ignore_index=True)

    save_orders(orders_df)

def cargo_tracking_main():
    global blockchain
    blockchain = Blockchain()

    try:
        sales_data = productorder.load_sales_data()
    except Exception as e:
        print(f"Failed to load sales data: {e}")
        return

    excess_inventory = productorder.calculate_excess_inventory(sales_data)

    product_name_input = input("Enter the product name: ")
    productorder.take_orders(product_name_input)
    
    try:
        orders_df = productorder.load_orders()
    except Exception as e:
        print(f"Failed to load orders: {e}")
        return

    orders_df['Shop'] = orders_df['Shop'].apply(productorder.rename_shop)

    try:
        ranked_stores = productorder.rank_stores('shop_sale.csv', 'shop_reviews.csv')
        ranked_stores = [(productorder.rename_shop(store), score) for store, score in ranked_stores]
        print("Ranking of stores:")
        for rank, (store, _) in enumerate(ranked_stores, start=1):
            print(f"{rank}. {store}")
    except Exception as e:
        print(f"Failed to rank stores: {e}")

    product_orders = orders_df[orders_df['Product Name'] == product_name_input]
    product_orders.loc[:, 'Order Quantity'] = product_orders['Order Quantity'].apply(productorder.parse_order_quantity)
    product_orders = product_orders[pd.notna(product_orders['Order Date']) & (product_orders['Order Quantity'] > 0)]
    
    if product_orders.empty:
        print(f"No valid orders placed for {product_name_input}.")
        return

    quick_order_enabled = input("Do you want to enable quick orders? (yes/no): ").lower() == 'yes'
    preorder_enabled = input("Do you want to enable preorders? (yes/no): ").lower() == 'yes'
    
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
                tracking_df = process_quick_order(product_name_input, shop, order_date, order_quantity, excess_inventory, blockchain)
            elif preorder_enabled:
                tracking_df = process_preorder(product_name_input, shop, order_date, order_quantity, excess_inventory, blockchain)
            else:
                print(f"Normal order for {product_name_input} in {shop} on {order_date} for {order_quantity} units")
                tracking_df = simulate_cargo_shipping(product_name_input, order_date, is_quick_order=False)

            tracking_data.append(tracking_df)

    if tracking_data:
        tracking_data_df = pd.concat(tracking_data, ignore_index=True)
        print("Full Tracking Data:")
        print(tracking_data_df)

    blockchain.save_chain('supply_chain_blockchain.encrypted')
    print("Encrypted blockchain has been saved to 'supply_chain_blockchain.encrypted'.")

if __name__ == '__main__':
    cargo_tracking_main()
