import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

file_names = ["day1.csv", "day2.csv", "day3.csv", "day4.csv", "day5.csv", "day6.csv", "day7.csv"]
combine_df = pd.concat([pd.read_csv(file, encoding='latin1') for file in file_names], ignore_index=True)


combine_df.dropna(inplace=True)


reorder_threshold = {
    "Sofa": 4,
    "Television": 2,
    "Bed": 2,
    "Toaster": 2,
    "Coffee Maker": 2,
    "T-Shirt": 5,
    "Laptop": 2,
    "Dining Table": 1,
    "Refrigerator": 1,
    "Chair": 4,
    "Microwave": 2,
    "Washing Machine": 1,
    "Smartphone": 6,
    "Headphones": 7,
    "Blender": 3,
    "Monitor": 4,
    "Tablet": 5,
    "Camera": 2,
    "Vacuum Cleaner": 1,
    "Bookshelf": 2
}

move_to_visible_threshold = 2

scaler_demand = MinMaxScaler(feature_range=(0, 1))
scaler_stock = MinMaxScaler(feature_range=(0, 1))

def preprocessing_data(df):
    df['Scaled Demand'] = scaler_demand.fit_transform(df[['Amount Sold']])
    df['Scaled Visible Stock'] = scaler_stock.fit_transform(df[['Visible Stock']])
    df['Scaled Inventory Stock'] = scaler_stock.fit_transform(df[['Inventory']])
    return df

combine_df = preprocessing_data(combine_df)


sequence_length = 7
x, y = [], []
for i in range(len(combine_df) - sequence_length):
    x.append(combine_df[['Scaled Demand', 'Scaled Visible Stock', 'Scaled Inventory Stock']].iloc[i:i+sequence_length].values)
    y.append(combine_df['Scaled Demand'].iloc[i+sequence_length])
X, y = np.array(x), np.array(y)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(sequence_length, 3)),
    Dropout(0.2),
    LSTM(units=64),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])


def make_predictions(product_data, future_days=7):
    predictions = []
    for i in range(future_days):
        if len(product_data) < sequence_length:
            print("Insufficient data for making predictions.")
            break
        last_sequence = product_data[['Scaled Demand', 'Scaled Visible Stock', 'Scaled Inventory Stock']].iloc[-sequence_length:].values
        if last_sequence.shape == (sequence_length, 3):
            prediction = model.predict(last_sequence.reshape(1, sequence_length, 3)).flatten()[0]
        else:
            continue  
        scaled_prediction = scaler_demand.inverse_transform([[prediction]])[0][0]
        predictions.append(round(scaled_prediction))
        
        new_row = {
            'Scaled Demand': prediction,
            'Scaled Visible Stock': product_data['Scaled Visible Stock'].iloc[-1],
            'Scaled Inventory Stock': product_data['Scaled Inventory Stock'].iloc[-1]
        }
        product_data = pd.concat([product_data, pd.DataFrame([new_row])], ignore_index=True)
        
    return predictions

def check_reorder_and_print(product_data, product_name, predictions):
    reorder_thresh = reorder_threshold[product_name]
    for i, prediction in enumerate(predictions, start=1):
        future_date = pd.to_datetime(combine_df['Date'].max()) + pd.Timedelta(days=i)
        future_visible_stock = product_data['Visible Stock'].iloc[-1]
        future_inventory_stock = product_data['Inventory'].iloc[-1]
        if future_visible_stock < reorder_thresh or future_inventory_stock < reorder_thresh:
            print(f"Stock level is low (Visible: {int(future_visible_stock)} units, Inventory: {int(future_inventory_stock)} units) for {product_name} on {future_date.date()}")
            print("Send an order for restocking.")


def move_to_visible(product_data, product_name):
    last_visible_stock = product_data['Visible Stock'].iloc[-1]
    last_inventory_stock = product_data['Inventory'].iloc[-1]
    if last_visible_stock < move_to_visible_threshold and last_inventory_stock > 0:
        move_units = min(move_to_visible_threshold - last_visible_stock, last_inventory_stock)
        product_data.at[len(product_data) - 1, 'Visible Stock'] += move_units
        product_data.at[len(product_data) - 1, 'Inventory'] -= move_units

product_name_input = input("Enter the product name: ")

for product_name in combine_df['Product Name'].unique():
    if product_name == product_name_input:
        product_data = combine_df[combine_df['Product Name'] == product_name].reset_index(drop=True)
        product_data = preprocessing_data(product_data)

        print(f"Predictive Demand for {product_name}:")
        predictions = make_predictions(product_data)
        for i, prediction in enumerate(predictions, start=1):
            print(f"  {pd.to_datetime(combine_df['Date'].max()) + pd.Timedelta(days=i)} - {prediction}")

        if predictions:
            check_reorder_and_print(product_data, product_name, predictions)
        else:
            print("Not enough data to predict future demand.")
            
        move_to_visible(product_data, product_name)
        break
