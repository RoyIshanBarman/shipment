import pandas as pd
import finalstore1
import finalstore2
import finalstore3

def load_orders():
    try:
        orders_df = pd.read_csv('orders.csv', parse_dates=['Order Date'])
        return orders_df
    except FileNotFoundError:
        print("No orders found, starting with an empty order list.")
        return pd.DataFrame(columns=['Product Name', 'Shop', 'Order Date', 'Order Quantity'])

def save_orders(orders_df):
    orders_df.to_csv('orders.csv', index=False)

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

if __name__ == "__main__":
    product_name_input = input("Enter the product name: ")
    take_orders(product_name_input)
