import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import hashlib
import time
import json
from cryptography.fernet import Fernet
import base64

# Load data
sales_data = pd.read_csv('shop_sale.csv')
reviews_data = pd.read_csv('shop_reviews.csv')

# Print columns for debugging
print("Sales Data Columns:", sales_data.columns)
print("Reviews Data Columns:", reviews_data.columns)

# Replace spaces with underscores in Shop_ID
sales_data['Shop_ID'] = sales_data['Shop_ID'].str.replace(' ', '_')
reviews_data['Shop_ID'] = reviews_data['Shop_ID'].str.replace(' ', '_')

# Merge data on Shop_ID
merged_data = pd.merge(sales_data, reviews_data, on='Shop_ID')

# Print columns of merged data for debugging
print("Merged Data Columns:", merged_data.columns)

# Check if 'Total_Sales_Amount' column is present
if 'Total_Sales_Amount' not in merged_data.columns:
    raise ValueError("'Total_Sales_Amount' column is not found in the merged data. Please check the input CSV files.")

# Check if merged data is empty
if merged_data.empty:
    raise ValueError("The merged DataFrame is empty. Please check the input CSV files for consistency.")

# List of columns to drop
columns_to_drop = ['Shop_ID', 'Month', 'Total_Sales_Amount', 'Review Text', 'Review ID']

# Check and drop existing columns only
columns_to_drop = [col for col in columns_to_drop if col in merged_data.columns]
x = pd.get_dummies(merged_data.drop(columns=columns_to_drop, axis=1))
y = merged_data['Total_Sales_Amount']

# Check if feature set or target variable is empty
if x.empty or y.empty:
    raise ValueError("Feature set or target variable is empty. Please check the data processing steps.")

# Split data into training, validation, and test sets
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)

# Convert data to tensors
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define neural network model
class RankingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RankingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
input_size = x_train_tensor.shape[1]
hidden_size = 64
model = RankingModel(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
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

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(x_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor.unsqueeze(1))
print(f'Test Loss: {test_loss.item()}')

# Save and load model
torch.save(model.state_dict(), 'ranking_model.pth')
model = RankingModel(input_size, hidden_size)
model.load_state_dict(torch.load('ranking_model.pth'))

# Predict and rank stores
features = pd.get_dummies(merged_data.drop(columns=columns_to_drop, axis=1)).values
features_tensor = torch.tensor(features, dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_sales = model(features_tensor)

predicted_sales = predicted_sales.squeeze().tolist()
stores = merged_data['Shop_ID'].tolist()
store_sales_dict = dict(zip(stores, predicted_sales))

ranked_stores = sorted(store_sales_dict.items(), key=lambda x: x[1], reverse=True)

print("Ranking of stores based on predicted sales:")

#for the storage and inplementation of chain
ranked_store=[]
'''for rank, (store, _) in enumerate(ranked_stores, start=1):
    print(f"{rank}. {store}")
    ranked_store.append(f"{rank}. {store}")'''
#print(ranked_store)

#Block chain part for the transaction

class Block:
    def __init__(self, index, previous_hash, timestamp, data, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        block_data = {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "data": self.data,
            "nonce": self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def __str__(self):
        block_data = {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "data": self.data,
            "nonce": self.nonce,
            "hash": self.hash
        }
        return json.dumps(block_data, sort_keys=True, indent=4)

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.difficulty = 4
        self.pending_transactions = []
        self.mining_reward = 100
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key) #generate the key
    
    def create_genesis_block(self):
        return Block(0, "0", time.time(), "Genesis Block")
    
    def get_latest_block(self):
        return self.chain[-1]
    
    def mine_pending_transactions(self, mining_reward_address):
        new_block = Block(len(self.chain), self.get_latest_block().hash, time.time(), self.pending_transactions)
        new_block = self.proof_of_work(new_block)
        self.chain.append(new_block)
        self.pending_transactions = [
            {"from": None, "to": mining_reward_address, "amount": self.mining_reward}
        ]
    
    def add_transaction(self, sender, item, quantity):
        if not sender or not item or not quantity:
            raise ValueError("Transaction must include sender, item, and quantity")
        
        transaction_hash = self.calculate_transaction_hash(sender, item, quantity, time.time())
        encrypted_transaction = self.encrypt_transaction(sender, item, quantity, transaction_hash)
        self.pending_transactions.append(encrypted_transaction)
    
    def calculate_transaction_hash(self, sender, item, quantity, timestamp):
        transaction_data = {
            "sender": sender,
            "item": item,
            "quantity": quantity,
            "timestamp": timestamp
        }
        transaction_string = json.dumps(transaction_data, sort_keys=True).encode()
        return hashlib.sha256(transaction_string).hexdigest()
    
    def encrypt_transaction(self, sender, item, quantity, transaction_hash):
        transaction = json.dumps({
            "sender": sender,
            "item": item,
            "quantity": quantity,
            "transaction_hash": transaction_hash
        }).encode()
        
        encrypted_transaction = self.cipher_suite.encrypt(transaction)
        return base64.urlsafe_b64encode(encrypted_transaction).decode()
    
    def proof_of_work(self, block):
        while block.hash[:self.difficulty] != "0" * self.difficulty:
            block.nonce += 1
            block.hash = block.calculate_hash()
        return block
    
    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True

class Store:
    def __init__(self, store_id):
        self.store_id = store_id
        self.orders = []
    
    def place_order(self, blockchain, item, quantity):
        blockchain.add_transaction(self.store_id, item, quantity)
        transaction_hash = blockchain.calculate_transaction_hash(self.store_id, item, quantity, time.time())
        self.orders.append({
            "item": item,
            "quantity": quantity,
            "transaction_hash": transaction_hash
        })


    # Assuming `ranked_stores` contains tuples like (store_name, store_object) and you have store_a, store_b, store_c defined


#ranked_store = []  # This list will hold the ranked stores with their orders

if __name__ == "__main__":
    # List of items
    items = [
        "Sofa",
        "Television",
        "Bed",
        "Toaster",
        "Coffee Maker",
        "T-Shirt",
        "Laptop",
        "Dining Table",
        "Refrigerator",
        "Chair",
        "Blender",
        "Jeans",
        "Smartphone",
        "Nightstand",
        "Microwave",
        "Shirt",
        "Tablet",
        "Desk Lamp",
        "Vacuum Cleaner",
        "Couch"
    ]

    # Initialize the blockchain
    my_blockchain = Blockchain()

    # Create stores
    store_a = Store("Store_A")
    store_b = Store("Store_B")
    store_c = Store("Store_C")

    # Stores place orders
    store_a.place_order(my_blockchain, "Sofa", 2)
    store_b.place_order(my_blockchain, "Television", 3)
    store_a.place_order(my_blockchain, "Bed", 1)
    store_b.place_order(my_blockchain, "Laptop", 4)
    store_a.place_order(my_blockchain, "Couch", 5)
    store_b.place_order(my_blockchain, "Shirt", 2)
    store_c.place_order(my_blockchain, "Bed", 5)
    store_c.place_order(my_blockchain, "Blender", 3)
    

    # Mine pending transactions
    print("Starting the miner...")
    my_blockchain.mine_pending_transactions("Miner1")

    # Validate the blockchain
    print("Blockchain valid?", my_blockchain.is_chain_valid())

    # Print the blockchain
    for block in my_blockchain.chain:
        print(block)
    temp_val=1
    for rank, (store, _) in enumerate(ranked_stores, start=1):
        print(f"{rank}. {store}")
        ranked_store.append(f"{rank}. {store}")
        '''
        if store in "A":
            print("Store A Orders:", store_a.orders)
            #temp_val+=1
        elif store in "B":
            print("Store A Orders:", store_b.orders)
            #temp_val+=1
        elif store in "C":
            print("Store A Orders:", store_c.orders)
            #temp_val+=1
        '''
    print(ranked_store)

    for i in range(len(ranked_store)):
        print(1)
        if "Shop_A" in ranked_store[i]:
            print("Store A Orders:", store_a.orders)
        elif "Shop_B" in ranked_store[i]:
            print("Store B Orders:", store_b.orders)
        elif "Shop_C" in ranked_store[i]:
            print("Store C Orders:", store_c.orders)