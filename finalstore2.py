import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

ReorderThresholds = {
    "Sofa": 15,
    "Television": 2,
    "Bed": 2,
    "Toaster": 2,
    "Coffee Maker": 2,
    "T-Shirt": 10,
    "Laptop": 2,
    "Dining Table": 1,
    "Refrigerator": 1,
    "Chair": 4,
    "Blender": 3,
    "Jeans": 10,
    "Smartphone": 6,
    "Nightstand": 3,
    "Microwave": 2,
    "Shirt": 10,
    "Tablet": 5,
    "Desk Lamp": 5,
    "Vacuum Cleaner": 1,
    "Couch": 5
}
move_to_visible_threshold = 2

def load_and_preprocess_data(file_names):
    combine_df = pd.concat([pd.read_csv(file, encoding='latin1', parse_dates=['Date'], dayfirst=True) for file in file_names], ignore_index=True)
    combine_df.dropna(inplace=True)
    return combine_df

def preprocessing_data(df):
    scaler_demand = MinMaxScaler(feature_range=(0, 1))
    scaler_stock = MinMaxScaler(feature_range=(0, 1))
    
    df['Scaled Demand'] = scaler_demand.fit_transform(df[['Amount Sold']])
    df['Scaled Visible Stock'] = scaler_stock.fit_transform(df[['Visible Stock']])
    df['Scaled Inventory Stock'] = scaler_stock.fit_transform(df[['Inventory']])
    return df, scaler_demand, scaler_stock

def create_lstm_model(sequence_length):
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=(sequence_length, 3)),
        Dropout(0.2),
        LSTM(units=64),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])

def make_predictions(model, product_data, scaler_demand, sequence_length, future_days=7):
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
        
        new_row = pd.DataFrame({
            'Scaled Demand': [prediction],
            'Scaled Visible Stock': [product_data['Scaled Visible Stock'].iloc[-1] - prediction],
            'Scaled Inventory Stock': [product_data['Scaled Inventory Stock'].iloc[-1] - prediction]
        }, index=[len(product_data)])
        product_data = pd.concat([product_data, new_row])
        
        # Update stock levels
        visible_stock_change = round(scaled_prediction)
        inventory_stock_change = round(scaled_prediction)
        product_data.at[len(product_data) - 1, 'Visible Stock'] = product_data['Visible Stock'].iloc[-2] - visible_stock_change
        product_data.at[len(product_data) - 1, 'Inventory'] = product_data['Inventory'].iloc[-2] - inventory_stock_change
        
    return predictions

def check_reorder_and_print(product_data, product_name, predictions):
    reorder_thresh = ReorderThresholds[product_name]
    for i, prediction in enumerate(predictions, start=1):
        future_date = pd.to_datetime(product_data['Date'].max(), dayfirst=True) + pd.Timedelta(days=i)
        future_visible_stock = product_data['Visible Stock'].iloc[-1] - prediction
        future_inventory_stock = product_data['Inventory'].iloc[-1] - prediction
        print(f"Debug: {future_date.date()} - Future Visible Stock: {future_visible_stock}, Future Inventory Stock: {future_inventory_stock}")
        if future_visible_stock < reorder_thresh and future_inventory_stock < reorder_thresh:
            print(f"Stock level is low (Visible: {int(future_visible_stock)} units, Inventory: {int(future_inventory_stock)} units) for {product_name} on {future_date.date()}")
            return True, future_date.date()
    return False, None

def move_to_visible(product_data, product_name):
    last_visible_stock = product_data['Visible Stock'].iloc[-1]
    last_inventory_stock = product_data['Inventory'].iloc[-1]
    if last_visible_stock < move_to_visible_threshold and last_inventory_stock > 0:
        move_units = min(move_to_visible_threshold - last_visible_stock, last_inventory_stock)
        product_data.at[len(product_data) - 1, 'Visible Stock'] += move_units
        product_data.at[len(product_data) - 1, 'Inventory'] -= move_units

def demand_forecasting_main(file_names, product_name_input):
    combine_df = load_and_preprocess_data(file_names)
    combine_df, scaler_demand, scaler_stock = preprocessing_data(combine_df)

    sequence_length = 7
    x, y = [], []
    for i in range(len(combine_df) - sequence_length):
        x.append(combine_df[['Scaled Demand', 'Scaled Visible Stock', 'Scaled Inventory Stock']].iloc[i:i+sequence_length].values)
        y.append(combine_df['Scaled Demand'].iloc[i+sequence_length])
    X, y = np.array(x), np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # LSTM model
    model = create_lstm_model(sequence_length)
    train_model(model, X_train, y_train, X_val, y_val)

    for product_name in combine_df['Product Name'].unique():
        if product_name == product_name_input:
            product_data = combine_df[combine_df['Product Name'] == product_name].reset_index(drop=True)
            product_data = preprocessing_data(product_data)[0]

            print(f"Predictive Demand for {product_name}:")
            predictions = make_predictions(model, product_data, scaler_demand, sequence_length)
            for i, prediction in enumerate(predictions, start=1):
                print(f"  {pd.to_datetime(combine_df['Date'].max()) + pd.Timedelta(days=i)} - {prediction}")

            if predictions:
                reorder_needed, reorder_date = check_reorder_and_print(product_data, product_name, predictions)
                if reorder_needed:
                    order_quantity = calculate_order_quantity(product_data, predictions, ReorderThresholds[product_name])
                    print(f"Order quantity needed: {order_quantity}")
                    return reorder_date, order_quantity
            else:
                print("Not enough data to predict future demand.")
            
            move_to_visible(product_data, product_name)
            break
    return None, 0

def calculate_order_quantity(product_data, predictions, reorder_threshold):
    future_visible_stock = product_data['Visible Stock'].iloc[-1]
    future_inventory_stock = product_data['Inventory'].iloc[-1]
    
    total_predicted_demand = sum(predictions)
    total_future_stock = future_visible_stock + future_inventory_stock

    if total_future_stock < total_predicted_demand:
        return total_predicted_demand - total_future_stock
    return 0

if __name__ == "__main__":
    file_names = ["shop_2.csv"]
    product_name_input = input("Enter the product name: ")
    reorder_date, order_quantity = demand_forecasting_main(file_names, product_name_input)
    if reorder_date:
        print(f"Reorder needed on {reorder_date} for {order_quantity} units.")
    else:
        print("No reorder needed.")
