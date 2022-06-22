import pandas as pd

class feature_extractor:
    def create_features(df):
        """
        Creates time series features from datetime index
        """
        #df['ds'] = pd.to_datetime(df['ds']).dt.date
        try:
            df = df.set_index('ds')
        except:
            pass
        
        df['date'] = pd.to_datetime(df.index, errors='coerce')

        
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear

        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        
        X = df[['y', 'dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear',
            'hour', 'minute']]
        
        X.index=df.index
        return X

    def series_lagger(data, full_data, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('lag%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('lag%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('lag%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        #print(agg.columns[:-1])
        for i in agg.columns[:-1]:
            full_data[i] = agg[i]
        return full_data[n_in:]
