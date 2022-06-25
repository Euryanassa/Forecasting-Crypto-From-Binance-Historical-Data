from pycaret.regression import load_model, predict_model
from preprocessor import preprocessing
from feature_extractor import feature_extractor
import numpy as np
import pandas as pd
import argparse


def forecastor(df, model, forecast_range_min = 10, num_of_lag = 10):
    '''
    forecastor(df = df,
           model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Models_Deneme/omp_model_2022_6_18-11:33:24',
           forecast_range_min = 10,
           num_of_lag = 10)
    '''

    # Dates and Forecasts to be appended
    try:
        print(f'[\033[96mLOG\033[0m]Loading Model {model}')
        model = load_model(model[:-4])
        print(f'[\033[92mSUCCESS\033[0m]Model Loaded Successfully')
    except:
        print(f'[\033[91mERROR\033[0m]Model Couldn\'t be Obtained Properly')
        print(f'[\033[93mHELP\033[0m]Check Your Model Path')
        return


    try:
        print(f'[\033[96mLOG\033[0m]Obtaining Data')
        last_df = df.tail(1).copy()
        dates_of_forecast = np.array([])
        forecasts = np.array([])
        print(f'[\033[92mSUCCESS\033[0m]Data Successfully Obtained')
    except:
        print(f'[\033[91mERROR\033[0m]Data Couldn\'t be Obtained')
        print(f'[\033[93mHELP\033[0m]Check Your Data Path')
        return

    try:
        print(f'[\033[96mLOG\033[0m]Forecasting Begining')
        for ff in range(forecast_range_min):

            # forecast one
            try:
                pred = predict_model(model, last_df)['Label'].values[0]
            except:
                print(f'[\033[91mERROR\033[0m]Forecast Error at Step {ff}')
                break

            # Lag shifting
            try:
                for i in range(num_of_lag, 1, -1):
                    last_df[f'lag1(t-{str(i)})'] = last_df[f'lag1(t-{str(i - 1)})']
                last_df['lag1(t-1)'] = pred
            except:
                print(f'[\033[91mERROR\033[0m]Lag Shifting Error at Step {ff} Inside Data')
                break

            # append prediction to array
            forecasts = np.append(forecasts, pred)

            try:
                # Adding one minute
                last_df.index = pd.to_datetime(last_df.index.astype(str)) + pd.DateOffset(minutes = 1)
                df_dater = feature_extractor.create_features(last_df)
            except:
                print(f'[\033[91mERROR\033[0m]Date Feature Creation Error at Step {ff} Inside Data')
                break


            try:
                for cols,dates in list(zip(df_dater.iloc[:,1:].columns, df_dater.iloc[:,1:].values[0])):
                    last_df[cols] = dates
            except:
                print(f'[\033[91mERROR\033[0m]Date Feature Append Error at Step {ff} with Forecast Data')
                break

            try:
                # append prediction to array
                dates_of_forecast = np.append(dates_of_forecast, df_dater.index.astype(str))
            except:
                print(f'[\033[91mERROR\033[0m]Append Error {ff} with Forecast Data')
                break


        print(f'[\033[92mSUCCESS\033[0m]Forecasting Successfully Done')

    except:
        print(f'[\033[91mERROR\033[0m]Couldn\'t Forecasted Properly')

    print(f'[\033[96mLOG\033[0m]Creating Forecast Dataframe')
    try:

        df_return = pd.DataFrame(np.array(list(zip(dates_of_forecast,forecasts))), columns = ['Dates', 'Forecasts'])
        df_return['Dates'] = pd.to_datetime(df_return['Dates'])
        print(f'[\033[92mSUCCESS\033[0m]Forecasting Dataframe Successfully Created')

    except:
        print(f'[\033[91mERROR\033[0m]Couldn\'t Create Foreacasting Dataframe!')
    
    return df_return

def parse_opt():
    parser = argparse.ArgumentParser()
    # df, model, forecast_range_min = 10, num_of_lag = 10
    parser.add_argument('--df', help = 'Path to your parquet file from binance dataset')
    parser.add_argument('--model', help = 'Path to your model')
    parser.add_argument('--forecast_range_min', default = 10, help = 'Minimum minute to be forecasted')
    parser.add_argument('--num_of_lag', default = 10, help = 'lag range which you trained on your model')
    opt = parser.parse_args()
    return opt

def main(df, model, forecast_range_min, num_of_lag):
    df = preprocessing.df_pp(df, freq = 'T')
    df = feature_extractor.create_features(df) 
    df = feature_extractor.series_lagger(df[['y']], full_data = df, n_in = 10, n_out=1, dropnan=True)
    
    df_out = forecastor(df, model, int(forecast_range_min), int(num_of_lag))

    print(df_out)


if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))
