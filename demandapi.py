from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)

reorder_threshold = {
    "Sofa": 15,
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

class DemandPredictionResponse(BaseModel):
    product_code: str
    daily_demand: List[int]
    demand_spike: str
    expected_demand_date_range: List[str]
    stock_depletion_datetime: Optional[str] = None
    remaining_stock: int
    order_quantity: Optional[int] = None
    message: Optional[str] = None

def load_and_preprocess_data(file_names):
    logging.info(f"Loading data from files: {file_names}")
    combine_df = pd.concat([pd.read_csv(file, encoding='latin1', parse_dates=['Date'], dayfirst=True) for file in file_names], ignore_index=True)
    combine_df.dropna(inplace=True)
    logging.info("Data loaded and preprocessed successfully.")
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
    current_stock = product_data['Visible Stock'].iloc[-1] + product_data['Inventory'].iloc[-1]
    initial_stock = current_stock
    stock_depletion_datetime = None
    last_date = pd.to_datetime(product_data['Date'].max())

    for i in range(future_days):
        if len(product_data) < sequence_length:
            break
        last_sequence = product_data[['Scaled Demand', 'Scaled Visible Stock', 'Scaled Inventory Stock']].iloc[-sequence_length:].values
        if last_sequence.shape == (sequence_length, 3):
            prediction = model.predict(last_sequence.reshape(1, sequence_length, 3)).flatten()[0]
        else:
            continue 
        scaled_prediction = int(scaler_demand.inverse_transform([[prediction]])[0][0])
        predictions.append(scaled_prediction)

        # Calculate more precise depletion time
        if current_stock > reorder_threshold[product_data['Product Name'].iloc[0]]:
            hours_to_deplete = ((current_stock - reorder_threshold[product_data['Product Name'].iloc[0]]) / scaled_prediction) * 24
            if hours_to_deplete < 24:
                depletion_time = last_date + pd.Timedelta(hours=hours_to_deplete)
                if stock_depletion_datetime is None:
                    stock_depletion_datetime = depletion_time
                current_stock = reorder_threshold[product_data['Product Name'].iloc[0]]
            else:
                current_stock -= scaled_prediction
        elif stock_depletion_datetime is None:
            stock_depletion_datetime = last_date

        last_date += pd.Timedelta(days=1)

        new_row = pd.DataFrame({
            'Scaled Demand': [prediction],
            'Scaled Visible Stock': [product_data['Scaled Visible Stock'].iloc[-1] - prediction],
            'Scaled Inventory Stock': [product_data['Scaled Inventory Stock'].iloc[-1] - prediction]
        }, index=[len(product_data)])
        product_data = pd.concat([product_data, new_row])
        
        visible_stock_change = round(scaled_prediction)
        inventory_stock_change = round(scaled_prediction)
        product_data.at[len(product_data) - 1, 'Visible Stock'] = max(0, product_data['Visible Stock'].iloc[-2] - visible_stock_change)
        product_data.at[len(product_data) - 1, 'Inventory'] = max(0, product_data['Inventory'].iloc[-2] - inventory_stock_change)
    
    remaining_stock = max(0, initial_stock - sum(predictions))
    return predictions, stock_depletion_datetime, remaining_stock

def move_to_visible(product_data, product_name):
    last_visible_stock = product_data['Visible Stock'].iloc[-1]
    last_inventory_stock = product_data['Inventory'].iloc[-1]
    if last_visible_stock < reorder_threshold[product_name] and last_inventory_stock > 0:
        move_units = min(reorder_threshold[product_name] - last_visible_stock, last_inventory_stock)
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

    model = create_lstm_model(sequence_length)
    train_model(model, X_train, y_train, X_val, y_val)

    if product_name_input not in combine_df['Product Name'].unique():
        logging.error(f"Product '{product_name_input}' not found in the dataset.")
        return None, None, 0
    
    product_data = combine_df[combine_df['Product Name'] == product_name_input].reset_index(drop=True)
    product_data = preprocessing_data(product_data)[0]

    predictions, stock_depletion_datetime, remaining_stock = make_predictions(model, product_data, scaler_demand, sequence_length)
    move_to_visible(product_data, product_name_input)
    
    if remaining_stock < reorder_threshold[product_name_input]:
        order_quantity = reorder_threshold[product_name_input] - remaining_stock
    else:
        order_quantity = 0

    return predictions, stock_depletion_datetime, remaining_stock, order_quantity

# FastAPI Routes
@app.get("/api/demand/store1/{product_code}", response_model=DemandPredictionResponse)
def get_demand_prediction_store1(product_code: str):
    file_names = ["shop_1_combined.csv"]  # Update with actual file paths for store 1
    predictions, stock_depletion_datetime, remaining_stock, order_quantity = demand_forecasting_main(file_names, product_code)
    if predictions:
        demand_spike = "High" if max(predictions) > 20 else "Medium" if max(predictions) > 10 else "Low"
        response = {
            "product_code": product_code,
            "daily_demand": predictions,
            "demand_spike": demand_spike,
            "expected_demand_date_range": [
                str(pd.to_datetime(pd.read_csv(file_names[0], encoding='latin1', parse_dates=['Date'], dayfirst=True)['Date'].max()) + pd.Timedelta(days=i)) for i in range(len(predictions))
            ],
            "stock_depletion_datetime": str(stock_depletion_datetime) if stock_depletion_datetime else None,
            "remaining_stock": remaining_stock,
            "order_quantity": order_quantity,
            "message": "No reorder required" if order_quantity == 0 else None
        }
        return response
    return {
        "product_code": product_code,
        "daily_demand": [],
        "demand_spike": "No Data",
        "expected_demand_date_range": [],
        "stock_depletion_datetime": None,
        "remaining_stock": 0,
        "order_quantity": 0,
        "message": "Product not found"
    }

@app.get("/api/demand/store2/{product_code}", response_model=DemandPredictionResponse)
def get_demand_prediction_store2(product_code: str):
    file_names = ["shop_2.csv"]  # Update with actual file paths for store 2
    predictions, stock_depletion_datetime, remaining_stock, order_quantity = demand_forecasting_main(file_names, product_code)
    if predictions:
        demand_spike = "High" if max(predictions) > 20 else "Medium" if max(predictions) > 10 else "Low"
        response = {
            "product_code": product_code,
            "daily_demand": predictions,
            "demand_spike": demand_spike,
            "expected_demand_date_range": [
                str(pd.to_datetime(pd.read_csv(file_names[0], encoding='latin1', parse_dates=['Date'], dayfirst=True)['Date'].max()) + pd.Timedelta(days=i)) for i in range(len(predictions))
            ],
            "stock_depletion_datetime": str(stock_depletion_datetime) if stock_depletion_datetime else None,
            "remaining_stock": remaining_stock,
            "order_quantity": order_quantity,
            "message": "No reorder required" if order_quantity == 0 else None
        }
        return response
    return {
        "product_code": product_code,
        "daily_demand": [],
        "demand_spike": "No Data",
        "expected_demand_date_range": [],
        "stock_depletion_datetime": None,
        "remaining_stock": 0,
        "order_quantity": 0,
        "message": "Product not found"
    }

@app.get("/api/demand/store3/{product_code}", response_model=DemandPredictionResponse)
def get_demand_prediction_store3(product_code: str):
    file_names = ["shop_3.csv"]  # Update with actual file paths for store 3
    predictions, stock_depletion_datetime, remaining_stock, order_quantity = demand_forecasting_main(file_names, product_code)
    if predictions:
        demand_spike = "High" if max(predictions) > 20 else "Medium" if max(predictions) > 10 else "Low"
        response = {
            "product_code": product_code,
            "daily_demand": predictions,
            "demand_spike": demand_spike,
            "expected_demand_date_range": [
                str(pd.to_datetime(pd.read_csv(file_names[0], encoding='latin1', parse_dates=['Date'], dayfirst=True)['Date'].max()) + pd.Timedelta(days=i)) for i in range(len(predictions))
            ],
            "stock_depletion_datetime": str(stock_depletion_datetime) if stock_depletion_datetime else None,
            "remaining_stock": remaining_stock,
            "order_quantity": order_quantity,
            "message": "No reorder required" if order_quantity == 0 else None
        }
        return response
    return {
        "product_code": product_code,
        "daily_demand": [],
        "demand_spike": "No Data",
        "expected_demand_date_range": [],
        "stock_depletion_datetime": None,
        "remaining_stock": 0,
        "order_quantity": 0,
        "message": "Product not found"
    }
