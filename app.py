from flask import Flask, jsonify,request, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

import numpy as np
import pickle
import sklearn

import pandas as pd
import folium
import matplotlib.pyplot as plt
import seaborn as sns

#loading models



dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))





if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAInM7kCmXUlF9l7x0wwnE1jl12w3EPh30"

# Instantiate the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,  # Balance creativity and factual correctness
    timeout=30,       # Allow sufficient time for complex responses
    max_retries=3,   # Retry mechanism in case of failures
)
def fetch_data_from_file():
    # Assuming data is stored in a CSV file named crop_yield_data.csv
    data = pd.read_csv('yield_df.csv')
    return data

print(sklearn.__version__)

def read_csv_data():
    df = pd.read_csv("yield_df.csv")
    # Assuming your CSV has columns like 'Year', 'Country', 'Yield', adjust this as per your actual CSV structure
    return df.to_dict(orient='records')


import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for Matplotlib




#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = prediction[0][0])
@app.route('/mapping', methods=['GET'])
def world_map():
    data = fetch_data_from_file()
    top_crops_by_area = data.groupby("Area", group_keys=False).apply(lambda group: group.nlargest(3, "hg/ha_yield")).reset_index(drop=True)
    m = folium.Map(location=[20.0, 0.0], zoom_start=2)

    countries = [
        {"name": "Albania", "coords": [41.1533, 20.1683]},
        {"name": "Algeria", "coords": [28.0339, 1.6596]},
        {"name": "Angola", "coords": [-11.2027, 17.8739]},
        {"name": "Argentina", "coords": [-38.4161, -63.6167]},
        {"name": "Armenia", "coords": [40.0691, 45.0382]},
        {"name": "Australia", "coords": [-25.2744, 133.7751]},
        {"name": "Austria", "coords": [47.5162, 14.5501]},
        {"name": "Azerbaijan", "coords": [40.1431, 47.5769]},
        {"name": "Bahamas", "coords": [25.0343, -77.3963]},
        {"name": "Bahrain", "coords": [25.9304, 50.6378]},
        {"name": "Bangladesh", "coords": [23.685, 90.3563]},
        {"name": "Belarus", "coords": [53.7098, 27.9534]},
        {"name": "Belgium", "coords": [50.5039, 4.4699]},
        {"name": "Botswana", "coords": [-22.3285, 24.6849]},
        {"name": "Brazil", "coords": [-14.235, -51.9253]},
        {"name": "Bulgaria", "coords": [42.7339, 25.4858]},
        {"name": "Burkina Faso", "coords": [12.2383, -1.5616]},
        {"name": "Burundi", "coords": [-3.3731, 29.9189]},
        {"name": "Cameroon", "coords": [7.3697, 12.3547]},
        {"name": "Canada", "coords": [56.1304, -106.3468]},
        {"name": "Central African Republic", "coords": [6.6111, 20.9394]},
        {"name": "Chile", "coords": [-35.6751, -71.543]},
        {"name": "Colombia", "coords": [4.5709, -74.2973]},
        {"name": "Croatia", "coords": [45.1, 15.2]},
        {"name": "Denmark", "coords": [56.2639, 9.5018]},
        {"name": "Dominican Republic", "coords": [18.7357, -70.1627]},
        {"name": "Ecuador", "coords": [-1.8312, -78.1834]},
        {"name": "Egypt", "coords": [26.8206, 30.8025]},
        {"name": "El Salvador", "coords": [13.7942, -88.8965]},
        {"name": "Eritrea", "coords": [15.1794, 39.7823]},
        {"name": "Estonia", "coords": [58.5953, 25.0136]},
        {"name": "Finland", "coords": [61.9241, 25.7482]},
        {"name": "France", "coords": [46.6034, 1.8883]},
        {"name": "Germany", "coords": [51.1657, 10.4515]},
        {"name": "Ghana", "coords": [7.9465, -1.0232]},
        {"name": "Greece", "coords": [39.0742, 21.8243]},
        {"name": "Guatemala", "coords": [15.7835, -90.2308]},
        {"name": "Guinea", "coords": [9.9456, -9.6966]},
        {"name": "Guyana", "coords": [4.8604, -58.9302]},
        {"name": "Haiti", "coords": [18.9712, -72.2852]},
        {"name": "Honduras", "coords": [15.2, -86.2419]},
        {"name": "Hungary", "coords": [47.1625, 19.5033]},
        {"name": "India", "coords": [20.5937, 78.9629]},
        {"name": "Indonesia", "coords": [-0.7893, 113.9213]},
        {"name": "Iraq", "coords": [33.2232, 43.6793]},
        {"name": "Ireland", "coords": [53.1424, -7.6921]},
        {"name": "Italy", "coords": [41.8719, 12.5674]},
        {"name": "Jamaica", "coords": [18.1096, -77.2975]},
        {"name": "Japan", "coords": [36.2048, 138.2529]},
        {"name": "Kazakhstan", "coords": [48.0196, 66.9237]},
        {"name": "Kenya", "coords": [-0.0236, 37.9062]},
        {"name": "Latvia", "coords": [56.8796, 24.6032]},
        {"name": "Lebanon", "coords": [33.8547, 35.8623]},
        {"name": "Lesotho", "coords": [-29.61, 28.2336]},
        {"name": "Libya", "coords": [26.3351, 17.2283]},
        {"name": "Lithuania", "coords": [55.1694, 23.8813]},
        {"name": "Madagascar", "coords": [-18.7669, 46.8691]},
        {"name": "Malawi", "coords": [-13.2543, 34.3015]},
        {"name": "Malaysia", "coords": [4.2105, 101.9758]},
        {"name": "Mali", "coords": [17.5707, -3.9962]},
        {"name": "Mauritania", "coords": [21.0079, -10.9408]},
        {"name": "Mauritius", "coords": [-20.3484, 57.5522]},
        {"name": "Mexico", "coords": [23.6345, -102.5528]},
        {"name": "Montenegro", "coords": [42.7087, 19.3744]},
        {"name": "Morocco", "coords": [31.7917, -7.0926]},
    {"name": "Mozambique", "coords": [-18.6657, 35.5296]},
    {"name": "Namibia", "coords": [-22.9576, 18.4904]},
    {"name": "Nepal", "coords": [28.3949, 84.124]},
    {"name": "Netherlands", "coords": [52.1326, 5.2913]},
    {"name": "New Zealand", "coords": [-40.9006, 174.886]},
    {"name": "Nicaragua", "coords": [12.8654, -85.2072]},
    {"name": "Niger", "coords": [17.6078, 8.0817]},
    {"name": "Norway", "coords": [60.472, 8.4689]},
    {"name": "Pakistan", "coords": [30.3753, 69.3451]},
    {"name": "Papua New Guinea", "coords": [-6.314993, 143.95555]},
    {"name": "Peru", "coords": [-9.19, -75.0152]},
    {"name": "Poland", "coords": [51.9194, 19.1451]},
    {"name": "Portugal", "coords": [39.3999, -8.2245]},
    {"name": "Qatar", "coords": [25.3548, 51.1839]},
    {"name": "Romania", "coords": [45.9432, 24.9668]},
    {"name": "Rwanda", "coords": [-1.9403, 29.8739]},
    {"name": "Saudi Arabia", "coords": [23.8859, 45.0792]},
    {"name": "Senegal", "coords": [14.4974, -14.4524]},
    {"name": "Slovenia", "coords": [46.1512, 14.9955]},
    {"name": "South Africa", "coords": [-30.5595, 22.9375]},
    {"name": "Spain", "coords": [40.4637, -3.7492]},
    {"name": "Sri Lanka", "coords": [7.8731, 80.7718]},
    {"name": "Sudan", "coords":[12.8628, 30.2176]},
    {"name": "Suriname", "coords": [3.9193, -56.0278]},
    {"name": "Sweden", "coords": [60.1282, 18.6435]},
    {"name": "Switzerland", "coords": [46.8182, 8.2275]},
    {"name": "Tajikistan", "coords": [38.861, 71.2761]},
    {"name": "Thailand", "coords": [15.870032, 100.992541]},
    {"name": "Tunisia", "coords": [33.8869, 9.5375]},
    {"name": "Turkey", "coords": [38.9637, 35.2433]},
    {"name": "Uganda", "coords": [1.3733, 32.2903]},
    {"name": "Ukraine", "coords": [48.3794, 31.1656]},
    {"name": "United Kingdom", "coords": [55.3781, -3.436]},
    {"name": "Uruguay", "coords": [-32.5228, -55.7658]},
    {"name": "Zambia", "coords": [-13.133897, 27.849332]},
    {"name": "Zimbabwe", "coords": [-19.015438, 29.154857]},
    {"name": "Costa Rica", "coords": [9.7489, -83.7534]},
    {"name": "Cuba", "coords": [21.5218, -77.7812]},
    {"name": "Cyprus", "coords": [35.1264, 33.4299]},
    {"name": "Czech Republic", "coords": [49.8175, 15.473]},
    {"name": "Democratic Republic of the Congo", "coords": [-4.0383, 21.7587]},
    {"name": "Djibouti", "coords": [11.8251, 42.5903]},
    {"name": "East Timor", "coords": [-8.8742, 125.7275]},
    {"name": "Eswatini", "coords": [-26.5225, 31.4659]},
    {"name": "Ethiopia", "coords": [9.145, 40.4897]},
    {"name": "Fiji", "coords": [-17.7134, 178.065]},
    {"name": "Gabon", "coords": [-0.8037, 11.6094]},
    {"name": "Georgia", "coords": [42.3154, 43.3569]},
    {"name": "Ghana", "coords": [7.9465, -1.0232]},
    {"name": "Greece", "coords": [39.0742, 21.8243]},
    {"name": "Guatemala", "coords": [15.7835, -90.2308]},
    {"name": "Guinea", "coords": [9.9456, -9.6966]},
    {"name": "Guyana", "coords": [4.8604, -58.9302]},
    {"name": "Haiti", "coords": [18.9712, -72.2852]},
    {"name": "Honduras", "coords": [15.2, -86.2419]},
    {"name": "Hungary", "coords": [47.1625, 19.5033]},
    {"name": "Iceland", "coords": [64.9631, -19.0208]},
    {"name": "Iran", "coords": [32.4279, 53.688]},
    {"name": "Israel", "coords": [31.0461, 34.8516]},
    {"name": "Ivory Coast", "coords": [7.54, -5.5471]},
    {"name": "Jamaica", "coords": [18.1096, -77.2975]},
    {"name": "Jordan", "coords": [30.5852, 36.2384]},
    {"name": "Kuwait", "coords": [29.3759, 47.977]},
    {"name": "Kyrgyzstan", "coords": [41.2044, 74.7661]},
    {"name": "Laos", "coords": [19.8563, 102.4955]},
    {"name": "Liberia", "coords": [6.4281, -9.4295]},
    {"name": "Luxembourg", "coords": [49.8153, 6.1296]},
    {"name": "Madagascar", "coords": [-18.7669, 46.8691]},
    {"name": "Malaysia", "coords": [4.2105, 101.9758]},
    {"name": "Mali", "coords": [17.5707, -3.9962]},
    {"name": "Malta", "coords": [35.9375, 14.3754]},
    {"name": "Mauritania", "coords": [21.0079, -10.9408]},
    {"name": "Mauritius", "coords": [-20.3484, 57.5522]},
    {"name": "Moldova", "coords": [47.4116, 28.3699]},
    {"name": "Mongolia", "coords": [46.8625, 103.8467]},
    {"name": "Montenegro", "coords": [42.7087, 19.3744]},
    {"name": "Morocco", "coords": [31.7917, -7.0926]},
    {"name": "Mozambique", "coords": [-18.6657, 35.5296]},
    {"name": "Myanmar", "coords": [21.9162, 95.956]},
    {"name": "Namibia", "coords": [-22.9576, 18.4904]},
    {"name": "Nauru", "coords": [-0.5228, 166.9315]},
    {"name": "Nepal", "coords": [28.3949, 84.124]},
    {"name": "New Zealand", "coords": [-40.9006, 174.886]},
    {"name": "Nicaragua", "coords": [12.8654, -85.2072]},
    {"name": "Niger", "coords": [17.6078, 8.0817]},
    {"name": "Nigeria", "coords": [9.082, 8.6753]},
    {"name": "North Korea", "coords": [40.3399, 127.5101]},
    {"name": "North Macedonia", "coords": [41.6086, 21.7453]},
    {"name": "Norway", "coords": [60.472, 8.4689]},
    {"name": "Oman", "coords": [21.4735, 55.9754]},
    {"name": "Pakistan", "coords": [30.3753, 69.3451]},
    {"name": "Palau", "coords": [7.515, 134.5825]},
    {"name": "Panama", "coords": [8.5379, -80.7821]},
    {"name": "Papua New Guinea", "coords": [-6.314993, 143.95555]},
    {"name": "Paraguay", "coords": [-23.4425, -58.4438]},
    {"name": "Peru", "coords": [-9.19, -75.0152]},
    {"name": "Philippines", "coords": [12.8797, 121.774]},
    {"name": "Poland", "coords": [51.9194, 19.1451]},
    {"name": "Portugal", "coords": [39.3999, -8.2245]},
    {"name": "Qatar", "coords": [25.3548, 51.1839]},
    {"name": "Republic of the Congo", "coords": [-0.228, 15.8277]},
    {"name": "Romania", "coords": [45.9432, 24.9668]},
    {"name": "Russia", "coords": [61.524, 105.3188]},
    {"name": "Rwanda", "coords": [-1.9403, 29.8739]},
    {"name": "Saint Kitts and Nevis", "coords": [17.357822, -62.782998]},
    {"name": "Saint Lucia", "coords": [13.9094, -60.9789]},
    {"name": "Saint Vincent and the Grenadines", "coords": [12.9843, -61.2872]},
    {"name": "Samoa", "coords": [-13.759, -172.1046]},
    {"name": "San Marino", "coords": [43.9424, 12.4578]},
    {"name": "Saudi Arabia", "coords": [23.8859, 45.0792]},
    {"name": "Senegal", "coords": [14.4974, -14.4524]},
    {"name": "Serbia", "coords": [44.0165, 21.0059]},
    {"name": "Seychelles", "coords": [-4.6796, 55.492]},
    {"name": "Sierra Leone", "coords": [8.460555, -11.779889]},
    {"name": "Singapore", "coords": [1.3521, 103.8198]},
    {"name": "Slovakia", "coords": [48.669, 19.699]},
    {"name": "Slovenia", "coords": [46.1512, 14.9955]},
    {"name": "Solomon Islands", "coords": [-9.6457, 160.1562]},
    {"name": "Somalia", "coords": [5.152149, 46.199616]},
    {"name": "South Africa", "coords": [-30.5595, 22.9375]},
    {"name": "South Korea", "coords": [35.9078, 127.7669]},
    {"name": "South Sudan", "coords": [7.8627, 30.2176]},
    {"name": "Spain", "coords": [40.4637, -3.7492]},
    {"name": "Sri Lanka", "coords": [7.8731, 80.7718]},
    {"name": "Sudan", "coords": [12.8628, 30.2176]},
    {"name": "Suriname", "coords": [3.9193, -56.0278]},
    {"name": "Sweden", "coords": [60.1282, 18.6435]},
    {"name": "Switzerland", "coords": [46.8182, 8.2275]},
    {"name": "Syria", "coords": [34.8021, 38.9968]},
    {"name": "Taiwan", "coords": [23.6978, 120.9605]},
    {"name": "Tajikistan", "coords": [38.861, 71.2761]},
    {"name": "Tanzania", "coords": [-6.369028, 34.888822]},
    {"name": "Thailand", "coords": [15.870032, 100.992541]},
    {"name": "Togo", "coords": [8.6195, 0.8248]},
    {"name": "Tonga", "coords": [-21.1784, -175.1982]},
    {"name": "Trinidad and Tobago", "coords": [10.6918, -61.2225]},
    {"name": "Tunisia", "coords": [33.8869, 9.5375]},
    {"name": "Turkey", "coords": [38.9637, 35.2433]},
    {"name": "Turkmenistan", "coords": [38.9697, 59.5563]},
    {"name": "Tuvalu", "coords": [-7.1095, 177.6493]},
    {"name": "Uganda", "coords": [1.3733, 32.2903]},
    {"name": "Ukraine", "coords": [48.3794, 31.1656]},
    {"name": "United Arab Emirates", "coords": [23.4241, 53.8478]},
    {"name": "United Kingdom", "coords": [55.3781, -3.436]},
    {"name": "United States of America", "coords": [37.0902, -95.7129]},
    {"name": "Uruguay", "coords": [-32.5228, -55.7658]},
    {"name": "Uzbekistan", "coords": [41.3775, 64.5853]},
    {"name": "Vanuatu", "coords": [-15.3767, 166.9592]},
    {"name": "Vatican City", "coords": [41.9029, 12.4534]},
    {"name": "Venezuela", "coords": [6.4238, -66.5897]},
    {"name": "Vietnam", "coords": [14.0583, 108.2772]},
    {"name": "Yemen", "coords": [15.5527, 48.5164]},
    {"name": "Zambia", "coords": [-13.133897, 27.849332]},
    {"name": "Zimbabwe", "coords": [-19.015438, 29.154857]},
] 



    areas = data["Area"].unique()

    popup_width = 175
    popup_height = 100
    area_crops_info = {}

    for area in areas:
        area_name = area
        area_crops = top_crops_by_area[top_crops_by_area["Area"] == area_name]
        unique_crops = []
        
        for index, row in area_crops.iterrows():
            crop_name = row["Item"]
            crop_yield = row["hg/ha_yield"]
            if crop_name not in [crop["name"] for crop in unique_crops]:
                unique_crops.append({"name": crop_name, "yield": crop_yield})
        
        popup_content = f"<div style='width: {popup_width}px; height: {popup_height}px;'><h3>{area_name}</h3><ul>"
        for crop in unique_crops:
            popup_content += f"<li><b>{crop['name']}: {crop['yield']}</b></li>"
        popup_content += "</ul></div>"
        
        area_crops_info[area_name] = popup_content
    
    for country in countries:
        coords = country["coords"]
        country_name = country["name"]
        if country_name in area_crops_info:
            popup_content = area_crops_info[country_name]
            folium.Marker(location=coords, popup=popup_content).add_to(m)
        else:
            folium.Marker(location=coords).add_to(m)  # Fallback if no popup content
        
    m.save("world_map.html")
    with open("world_map.html", "r") as f:
        map_html = f.read()
    
    return map_html

@app.route('/visualization')
def visualization():
    yearly_yield_path = create_yearly_yield_chart()
    crop_comparison_path = create_crop_comparison_chart()
    time_series_path = create_time_series_chart()
    return render_template('visualization.html',
                           yearly_yield_path=yearly_yield_path,
                           crop_comparison_path=crop_comparison_path,
                           time_series_path=time_series_path)
    
def create_crop_comparison_chart():
    yield_df = fetch_data_from_file()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Item', y='hg/ha_yield', data=yield_df, errorbar=None)
    plt.title('Crop-wise Yield Comparison')
    plt.xlabel('Crop')
    plt.ylabel('Yield (hg/ha)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_path = 'static/images/crop_comparison.png'
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def create_time_series_chart():
    yield_df = fetch_data_from_file()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Year', y='hg/ha_yield', hue='Item', data=yield_df, estimator='mean', errorbar=None)
    plt.title('Time Series Analysis of Crop Yield')
    plt.xlabel('Year')
    plt.ylabel('Yield (hg/ha)')
    plt.tight_layout()
    chart_path = 'static/images/time_series.png'
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def create_yearly_yield_chart():
    yield_df = fetch_data_from_file()
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Year', y='hg/ha_yield', hue='Item', data=yield_df, errorbar=None)
    plt.title('Yearly Crop Yield')
    plt.xlabel('Year')
    plt.ylabel('Yield (hg/ha)')
    plt.legend(title='Crop')
    plt.tight_layout()
    chart_path = 'static/images/yearly_yield.png'
    plt.savefig(chart_path)
    plt.close()
    return chart_path


# Route for the chatbot page
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'GET':
        return render_template('chatbot.html')
    elif request.method == 'POST':
        # Check if the request is JSON
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        # Get JSON data
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"response": "Please enter a question."}), 400
        
        try:
            response = ask_farming_question(query)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500



# Function to handle farmer queries
def ask_farming_question(query):
    """Handles farmer queries and returns a response."""
    try:
        response = chain.invoke({"query": query})
        return response.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create a prompt template tailored for farmer Q&A
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an agricultural assistant designed to answer farmers' questions. "
                "Provide helpful, accurate, and concise answers. If you don't know an answer, "
                "recommend trusted agricultural resources."
            ),
        ),
        ("human", "{query}"),
    ]
)

chain = prompt | llm

if __name__ == "__main__":
    app.run(debug=True)
