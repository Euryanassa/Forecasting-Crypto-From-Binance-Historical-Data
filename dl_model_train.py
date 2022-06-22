import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM




class model_train_pool:
    def facebook_neural_prophet(df, freq, to_forecast):

        '''
        facebook_neural_prophet(df, freq = 'T', to_forecast = 10)
        '''

        try:
            m = NeuralProphet(daily_seasonality = True,
                            weekly_seasonality = False,
                            yearly_seasonality = False,
                            seasonality_mode='multiplicative',
                            learning_rate=0.01,
                            batch_size = 12,
                            n_forecasts = to_forecast,
                            n_lags = 1,
                            epochs = 12
                            )

            print('[INFO] Model No: 1 / Neural Prophet AR-Net')
            m.fit(df, freq = freq)
            print('[SUCCESS] Neural Prophet Model Trained Successfully')
            # MODEL SAVELE
            #torch.save(m, 'first_model.pt')
        except:
            print('[ERROR:002] Data Can Not Fitted Neural Prophet Model Properly')

        try:
            future = m.make_future_dataframe(df)
            forecast_n = m.predict(future, decompose = True, raw = True)
            pred = forecast_n.iloc[-1,1:to_forecast + 1].values
            
            print('[SUCCESS] Forecasted Successfully')
            return pred
        except:
            print('[ERROR:003] Can Not Inferance Neural Prophet Model Properly')

    def LSTM_Classic(df):
        #df = df_pp('BTC-EUR.parquet', freq = 'T')
        #df = create_features(df)
        #df = series_lagger(df[['y']], full_data = df, n_in = 10, n_out=1, dropnan=True)

        # split into train and test sets
        try:
            values = df.values
            n_train_hours = -int(df.shape[0]*0.05)
            train = values[:n_train_hours, :]
            test = values[n_train_hours:, :]
        

            # split into input and outputs
            train_X, train_y = train[:, 1:], train[:, 0]
            test_X, test_y = test[:, 1:], test[:, 0]

            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
            print('[SUCCESS] LSTM Train Test Splitted Successfully')

        except:
            print('[ERROR:004] LSTM Data Preprocessing Failed')


        # design network
        try:
            print('[INFO] Model No: 2 / LSTM')
            model = Sequential()
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer='adam')
            # fit network
            history = model.fit(train_X, train_y, epochs=3, batch_size=72, validation_data = (test_X, test_y), verbose = 1, shuffle=False)
            print('[SUCCESS] LSTM Train Has Finished Successfully')
            # MODEL SAVELE
        except:
            print('[ERROR:005] Data Can Not Fitted LSTM Properly')
        # plot history
        '''
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        '''
        # !!!! Simdilik testin tamamini donduruyor, duzeltilip, updatelenecek.
        # Laglanmis future list cikacak ve girilen inputa gore for loopunda forecast edecek
        try:
            yhat = model.predict(test_X) 
            print('[SUCCESS] Forecasted Successfully')
        except:
            print('[ERROR:006] Can Not Inferance LSTM Model Properly')

        return yhat