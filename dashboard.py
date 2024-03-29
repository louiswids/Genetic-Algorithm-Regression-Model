from os import pread
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import numpy as np

from sklearn.metrics import mean_squared_error 
import random
from datetime import datetime
from geneticalgorithm import geneticalgorithm as ga
import json
import time


# Functions
def PreProcessingData(file):

    data = json.load(file)



    # Creating lists for DataFrame columns
    dates = []
    sales = []
    prices = []


    # Extracting data and populating lists
    for item in data['prices']:
        date_str, price, sale = item
        # Parsing date and converting sales to integer
        dates.append(pd.to_datetime(date_str, format='%b %d %Y %H: +%f'))
        sales.append(int(sale))
        prices.append(float(price))

    # Creating a DataFrame
    df = pd.DataFrame({
        "Date": dates,
        "Number of Sales": sales,
        "Price": prices
    })
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate the total sales amount for each day
    df['Total Sales Amount'] = df['Number of Sales'] * df['Price']

    # Group the DataFrame by date, calculate total sales amount and total number of sales for each day
    result_df = df.groupby(df['Date'].dt.date).agg({'Total Sales Amount': 'sum', 'Number of Sales': 'sum'})

    # Calculate the average price per sale based on the formula
    result_df['Average Price per Sale'] = result_df['Total Sales Amount'] / result_df['Number of Sales']

    # Reset the index to get the 'Date' column back
    result_df = result_df.reset_index()

    # Set the display format for pandas to show numbers without scientific notation
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    
    # Set date as date-type data
    result_df['Date']=pd.to_datetime(result_df['Date'])
    
    result_df = result_df[['Date', 'Average Price per Sale']]
    
    return result_df

# Aggregate data weekly
def aggregate_weekly(data,start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    data['Date'] = pd.to_datetime(data['Date'])
    
    # Filter data based on date boundaries
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    # Set the 'Date' column as the index
    data.set_index('Date', inplace=True)
    
    # Resample the data to get weekly median
    weekly_median = data.resample('W').median()
    
     # Calculate the monthly percentage change
    weekly_median['Percentage Change'] = weekly_median['Average Price per Sale'].pct_change()
    
    # Fill NaN values with 0 for the first month
    weekly_median['Percentage Change'].fillna(0, inplace=True)
    
    return weekly_median

# Aggregate data biweekly
def aggregate_biweekly(data,start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Filter data based on date boundaries
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    # Set the 'Date' column as the index
    data.set_index('Date', inplace=True)
    
    # Resample the data to get biweekly median
    biweekly_median = data.resample('2W').median()
    
     # Calculate the monthly percentage change
    biweekly_median['Percentage Change'] = biweekly_median['Average Price per Sale'].pct_change()
    
    # Fill NaN values with 0 for the first month
    biweekly_median['Percentage Change'].fillna(0, inplace=True)
    
    return biweekly_median

def aggregate_monthly(data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    data['Date'] = pd.to_datetime(data['Date'])
    
    # Filter data based on date boundaries
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    # Set the 'Date' column as the index
    data.set_index('Date', inplace=True)
    
    # Resample the data to get monthly median
    monthly_median = data.resample('M').median()
    
    # Calculate the monthly percentage change
    monthly_median['Percentage Change'] = monthly_median['Average Price per Sale'].pct_change()
    
    # Fill NaN values with 0 for the first month
    monthly_median['Percentage Change'].fillna(0, inplace=True)
    
    return monthly_median



# Title
st.title("Genetic Algorithm Final Project")
st.subheader("Regression Model Parameters Identification")




uploaded_file = st.file_uploader("Choose a json file")
st.write("filename:", uploaded_file.name)
data = PreProcessingData(uploaded_file)
# Filtering Components
min_date = pd.to_datetime(data["Date"].min())
max_date = pd.to_datetime(data["Date"].max())
with st.sidebar:
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Time Range',
        min_value=min_date,  # Convert to date
        max_value=max_date, 
        value=[min_date, max_date]
    )
    timeFrame = st.selectbox(
        'Time Frame Aggregation',
        ['Monthly', 'Bi-Weekly','Weekly'],
        placeholder='Choose time aggregation'
    )
    number = st.number_input('Insert number of model parameter', min_value=1, step=1)



if timeFrame == 'Monthly':
    data = aggregate_monthly(data, start_date, end_date)
elif timeFrame == 'Bi-Weekly':
    data = aggregate_biweekly(data,start_date, end_date)
else:
    data = aggregate_weekly(data,start_date, end_date)

data

# Fitness Calculation
def calculate_fitness(params, real_data = data, nrows = number):
    # Initializing the predicted array
    pred_data = [0]*(len(real_data['Percentage Change']))
    
    # Storing previous data
    pred_data[0:nrows] = real_data['Percentage Change'][0:nrows]
    
    # Making the prediction
    y = 0.0
    for i in range(nrows, len(real_data['Percentage Change'])):
        y = 0.0
        for j in range(0,nrows):
            y = y + (params[j] * pred_data[i - nrows + j])
        pred_data[i - 1] = y
        
    # Calculating fitness
    error = mean_squared_error(real_data['Percentage Change'],pred_data) 
    
    return error

# Fitness Calculation
def predict(params, real_data = data, nrows = number):
    # Initializing the predicted array
    pred_data = [0]*len(real_data['Percentage Change'])
    
    # Storing previous data
    pred_data[0:nrows] = real_data['Percentage Change'][0:nrows]
    
    # Making the prediction
    y = 0.0
    for i in range(nrows, len(real_data['Percentage Change'])):
        y = 0.0
        for j in range(0, nrows):
            y = y + (params[j] * pred_data[i - nrows + j])
        pred_data[i - 1] = y
        
        
    return pred_data


fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x = 'Date', data = data, y = 'Percentage Change')
st.pyplot(fig)

varbound = np.array([[-1,1]]*number)      

algorithm_param = {'max_num_iteration': 100, 'population_size': 100, 'mutation_probability': 0.0125, 'elit_ratio': 0.01,
                    'crossover_probability': 0.7, 'parents_portion': 0.8, 'crossover_type': 'two_point',
                    'max_iteration_without_improv': 20}

# Create the genetic algorithm optimizer
model = ga(function=calculate_fitness, dimension=number, variable_type='real', variable_boundaries=varbound,
            algorithm_parameters=algorithm_param, function_timeout=120)

if st.button('Run GA!'):
    with st.spinner('Running Genetic Algorithm...'):
        model.run()
        m = model.best_variable
        predicted = predict(m)
        predicted = pd.DataFrame(predicted, columns=['Predicted'])
        result_df = pd.concat([data.reset_index(), predicted.reset_index()], axis = 1)
        result_df = result_df[['Date', 'Percentage Change', 'Predicted']]
        st.text("Best Model Parameter:\n"+str(m)+"\n")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=result_df)
        st.pyplot(fig)

# Footer
st.text("Bagus - Hylmi - Kreshna - Louis - Nadine - 2023")
