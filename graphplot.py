import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_names):
    dfs = []
    for file in file_names:
        df = pd.read_csv(file, encoding='latin1', parse_dates=['Date'], dayfirst=True)
        df['Shop'] = file.split('_')[1].split('.')[0]  # Extract shop number from filename
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_orders_by_shop(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Shop', y='Amount Sold', data=data)
    plt.title('Distribution of Orders by Shop')
    plt.xlabel('Shop')
    plt.ylabel('Amount Sold')
    plt.show()

def plot_top_items_demand(data, top_n=10):
    top_items = data.groupby('Product Name')['Amount Sold'].sum().nlargest(top_n).index
    top_items_data = data[data['Product Name'].isin(top_items)]
    
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Product Name', y='Amount Sold', hue='Shop', data=top_items_data)
    plt.title(f'Top {top_n} Items Demand by Shop')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Shop')
    plt.tight_layout()
    plt.show()

def plot_demand_over_time(data, top_n=5):
    top_items = data.groupby('Product Name')['Amount Sold'].sum().nlargest(top_n).index
    top_items_data = data[data['Product Name'].isin(top_items)]
    
    plt.figure(figsize=(14, 7))
    for item in top_items:
        item_data = top_items_data[top_items_data['Product Name'] == item]
        sns.lineplot(x='Date', y='Amount Sold', data=item_data, label=item)
    
    plt.title(f'Demand Over Time for Top {top_n} Items')
    plt.xlabel('Date')
    plt.ylabel('Amount Sold')
    plt.legend(title='Product')
    plt.tight_layout()
    plt.show()

def main():
    file_names = ["shop_1_combined.csv", "shop_2.csv", "shop_3.csv"]
    data = load_data(file_names)
    
    plot_orders_by_shop(data)
    plot_top_items_demand(data)
    plot_demand_over_time(data)

if __name__ == "__main__":
    main()