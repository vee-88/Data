from dataidea.packages import pd, plt, os, np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf


def matchTaxHeadNames(data):
    # drop rows will all data missing
    data.dropna(how='all', inplace=True)

    # handle for TAX HEAD NAME.1
    tax_head_name_1 = data[data['TAX HEAD NAME.1'] != '--']['TAX HEAD NAME.1']
    indexes = tax_head_name_1.index
    values = tax_head_name_1.values 

    # handle TAX HEAD NAME.2
    tax_head_name_2 = data[data['TAX HEAD NAME.2'] != '-']['TAX HEAD NAME.2']
    indexes = tax_head_name_2.index
    values = tax_head_name_2.values 

    # replace the data
    for count, index in enumerate(indexes):
        data.loc[index, 'TAX HEAD NAME'] = values[count]
    
    data.drop(columns=['TAX HEAD NAME.1', 'TAX HEAD NAME.2'], inplace=True)

    return data


# function to combine dataframes, takes in a list
def combineDataFrames(folder_paths=[], columns=[], output_file='combined_output.csv'):

    dataframes = []

    for path in folder_paths:
        for filename in os.listdir(path):
            if filename.endswith(".xls") or filename.endswith('.csv'):

                file_path = os.path.join(path, filename)

                df = pd.read_excel(file_path, header=2)
                
                selected_columns = df[columns].dropna()

                dataframes.append(selected_columns)

        combined_df = pd.concat(dataframes, ignore_index=True)

        combined_df.to_csv(output_file, index=False)

    return combined_df

# set indexes
def setIndex(dataframe, column):
    dataframe[column] = pd.to_datetime(dataframe[column])

    dataframe = dataframe.set_index(dataframe[column])
    dataframe.drop(column, axis = 1, inplace = True)

    return dataframe

# decomposer
def decompose(data):
    decomposition = seasonal_decompose(
        data, 
        model='additive')
    
    fig = decomposition.plot()
    
    return decomposition

# test for stationarity
def stationarity_test(timeseries):
    # Get rolling statistics for window = 12 i.e. yearly statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # Plot rolling statistic
    plt.figure(figsize= (10,6))
    plt.xlabel('Years')
    plt.ylabel('Bank Payment')    
    plt.title('Stationary Test: Rolling Mean and Standard Deviation')
    plt.plot(timeseries, color= 'blue', label= 'Original')
    plt.plot(rolling_mean, color= 'green', label= 'Rolling Mean')
    plt.plot(rolling_std, color= 'red', label= 'Rolling Std')   
    plt.legend()
    plt.show()
    
    # Dickey-Fuller test
    print('Results of Dickey-Fuller Test')
    df_test = adfuller(timeseries)
    df_output = pd.Series(df_test[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' %key] = value
    print(df_output)
