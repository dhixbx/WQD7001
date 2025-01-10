from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import os

# 初始化 Flask 应用
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# 加载训练好的模型
best_xgb = load('best_xgb_model.joblib')

# 定义预测函数
def predict_price(input_data):
    raw_new_data = pd.DataFrame(input_data)
    df3 = pd.read_csv('1.csv')

    # One-hot encoding
    new_data_encoded = pd.get_dummies(raw_new_data, columns=['Location', 'Property Type', 'Property_description'])
    columns_to_keep = ['Price', 'btm', 'bdm', 'Size']
    property_type_cols = [col for col in df3.columns if col.startswith('Property Type')]
    property_description_cols = [col for col in df3.columns if col.startswith('Property_description')]
    location_cols = [col for col in df3.columns if col.startswith('Location')]
    final_columns = columns_to_keep + property_type_cols + property_description_cols + location_cols

    # Align columns
    for col in set(final_columns) - set(new_data_encoded.columns):
        new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[final_columns[1:]]

    # Predict
    return best_xgb.predict(new_data_encoded)[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    predicted_prices = []
    growth_rates = [2.1243, 3.2643, 3.8809, 3.5014, 3.6112]
    predicted_price_2025 = None

    # 定义下拉框选项
    locations = [
        'Kulai', 'Johor Bahru', 'Iskandar Puteri', 'Permas Jaya', 'Senai', 'Tampoi',
        'Gelang Patah', 'Skudai', 'Masai', 'Muar', 'Pasir Gudang', 'Ulu Tiram',
        'Setia Indah', 'Setia Tropika', 'other', 'Perling', 'Kluang', 'Tebrau',
        'Batu Pahat', 'Kota Tinggi', 'Segamat'
    ]
    property_types = ['House', 'Apartment / Condominium']
    property_descriptions = [
        '2-storey Terraced House', 'Apartment', 'Flat', '1-storey Terraced House',
        'Cluster House', 'Service Residence', '3-storey Terraced House', 'Condominium',
        '1.5-storey Terraced House', 'Semi-Detached House', '2.5-storey Terraced House',
        'Terraced House', 'Townhouse', 'Others', 'Townhouse Condo', 'Bungalow House',
        'Studio', 'Duplex', 'Link Bungalow'
    ]

    if request.method == 'POST':
        # 获取表单数据
        size = float(request.form.get('size'))
        location = request.form.get('location')
        bdm = int(request.form.get('bdm'))
        btm = int(request.form.get('btm'))
        property_type = request.form.get('property_type')
        property_description = request.form.get('property_description')

        # 预测价格
        input_data = [{
            'Size': size,
            'Location': location,
            'bdm': bdm,
            'btm': btm,
            'Property Type': property_type,
            'Property_description': property_description
        }]
        predicted_price = predict_price(input_data)

        # 预测未来年份价格
        predicted_prices = [predicted_price]
        for rate in growth_rates:
            next_price = predicted_prices[-1] * (1 + rate / 100)
            predicted_prices.append(next_price)

        # 提取 2025 年的预测价格
        if len(predicted_prices) > 1:
            predicted_price_2025 = predicted_prices[1]  # 第二个值为 2025 年预测价格

        # 保存预测价格折线图
        years = list(range(2024, 2029))
        plt.figure(figsize=(10, 6))
        plt.plot(years, predicted_prices[:-1], marker='o', label="Predicted Prices")
        plt.title("Predicted Property Prices from 2024 to 2028")
        plt.xlabel("Year")
        plt.ylabel("Price (RM)")
        plt.grid(True)
        plt.legend()
        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig('static/predicted_price_chart.png')
        plt.close()

    return render_template('index.html',
                           predicted_price=predicted_price,
                           predicted_prices=predicted_prices,
                           predicted_price_2025=predicted_price_2025,
                           growth_rates=growth_rates,
                           locations=locations,
                           property_types=property_types,
                           property_descriptions=property_descriptions)

if __name__ == "__main__":
    app.run(debug=True)
