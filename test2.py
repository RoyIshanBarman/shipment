import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import finalstore1
import finalstore2
import finalstore3

def combine_csv_files(file_names):
    dataframes = [pd.read_csv(file) for file in file_names]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def simulate_cargo_shipping(product_name, order_date):
    delivery_time_mean = 48 
    delivery_time_std = 12  
    delivery_time = max(1, int(np.random.normal(delivery_time_mean, delivery_time_std)))
    
    # Convert order_date to datetime if it's a tuple
    if isinstance(order_date, tuple):
        order_date = datetime(*order_date)
    elif isinstance(order_date, str):
        order_date = datetime.strptime(order_date, "%Y-%m-%d")
    
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
def cargo_tracking_main():
    shop_files = {
        "shop1": ["day1.csv", "day2.csv", "day3.csv", "day4.csv", "day5.csv", "day6.csv", "day7.csv"],
        "shop2": ["shop_2.csv"],
        "shop3": ["shop_3.csv"]
    }
    
    shop_input = input("Enter the shop name (shop1, shop2, shop3): ").strip().lower()
    if shop_input not in shop_files:
        print("Invalid shop name. Please enter one of the following: shop1, shop2, shop3.")
        return
    
    product_name_input = input("Enter the product name: ").strip()


    shop_data = combine_csv_files(shop_files[shop_input])
    shop_combined_file = f"{shop_input}_combined.csv"
    shop_data.to_csv(shop_combined_file, index=False)


    reorder_date = None
    if shop_input == "shop1":
        reorder_date = finalstore1.demand_forecasting_main(shop_files[shop_input], product_name_input)
    elif shop_input == "shop2":
        reorder_date = finalstore2.demand_forecasting_main(shop_files[shop_input], product_name_input)
    elif shop_input == "shop3":
        reorder_date = finalstore3.demand_forecasting_main(shop_files[shop_input], product_name_input)

    if reorder_date:
        if isinstance(reorder_date, tuple):
            reorder_date, quantity = reorder_date
            reorder_date = datetime(*reorder_date)
            print(f"Reorder needed for {product_name_input} in {shop_input} on {reorder_date} for {quantity} units")
        else:
            print(f"Reorder needed for {product_name_input} in {shop_input} on {reorder_date}")
        
        tracking_df = simulate_cargo_shipping(product_name_input, reorder_date)
        print(tracking_df)
    else:
        print(f"No reorder needed for {product_name_input} in {shop_input}")

if __name__ == "__main__":
    cargo_tracking_main()
