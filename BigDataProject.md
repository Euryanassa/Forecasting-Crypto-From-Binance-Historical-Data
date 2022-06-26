# Preprocessing to for All Bionance Historical Dataset


```python
import pandas as pd
import pyarrow.parquet as pq

class preprocessing:
    def df_pp(path,freq):
        try:
            print(f'[\033[96mLOG\033[0m]Data {path} Reading')
            df = pq.read_table(source=path).to_pandas()
            df = df.iloc[-600:-200]
            print(f'[\033[92mSUCCESS\033[0m]Data Read Successfully')
        except:
            print(f'[\033[91mERROR\033[0m]Data Can Not Read {path} Successfully')
        try:
            print(f'[\033[96mLOG\033[0m]Data Conversion to {path} Begins.')
            df.rename(columns={'close': 'y'}, inplace=True, errors='raise')
            df['ds'] = df.index
            df.reset_index(drop=True, inplace=True)
            df = df[['ds','y']]
            date_range_in_minutes = pd.date_range(start = str(df.ds.values[0])[:19], end = str(df.ds.values[-1])[:19], freq=freq)
            date_df = pd.DataFrame(list(date_range_in_minutes), columns = ['ds'])
            date_df['ds'] = pd.to_datetime(date_df['ds']).astype('str')
            df['ds'] = pd.to_datetime(df['ds']).astype('str')
            all_date_df = date_df.merge(df, how = 'left',  on=["ds"])
            all_date_df = all_date_df.fillna(method='ffill')
            print(f'[\033[92mSUCCESS\033[0m]Data Converted Successfully')
            return all_date_df
        except:
            print(f'[\033[91mERROR\033[0m]Data Can Not Converted {path} Successfully')
```


```python
df = preprocessing.df_pp('/content/drive/MyDrive/BIG_DATA_PROJECT/Dataset/BTC-EUR.parquet', 'T')
```

    [[96mLOG[0m]Data /content/drive/MyDrive/BIG_DATA_PROJECT/Dataset/BTC-EUR.parquet Reading
    [[92mSUCCESS[0m]Data Read Successfully
    [[96mLOG[0m]Data Conversion to /content/drive/MyDrive/BIG_DATA_PROJECT/Dataset/BTC-EUR.parquet Begins.
    [[92mSUCCESS[0m]Data Converted Successfully



```python
df.tail()
```





  <div id="df-741d5971-bc60-42f0-a990-0e98f9b13d7d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>495</th>
      <td>2022-03-11 22:15:00</td>
      <td>35708.468750</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2022-03-11 22:16:00</td>
      <td>35712.550781</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2022-03-11 22:17:00</td>
      <td>35709.011719</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2022-03-11 22:18:00</td>
      <td>35722.570312</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2022-03-11 22:19:00</td>
      <td>35739.898438</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-741d5971-bc60-42f0-a990-0e98f9b13d7d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-741d5971-bc60-42f0-a990-0e98f9b13d7d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-741d5971-bc60-42f0-a990-0e98f9b13d7d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Extracting Future form Dataset and Historical Values as Lag for Forecasting


```python
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

```


```python
df = feature_extractor.create_features(df) 
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:23: FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.  Please use Series.dt.isocalendar().week instead.



```python
df.tail()
```





  <div id="df-d6478d6a-0880-42a2-b433-c998d8387973">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>dayofweek</th>
      <th>quarter</th>
      <th>month</th>
      <th>year</th>
      <th>dayofyear</th>
      <th>dayofmonth</th>
      <th>weekofyear</th>
      <th>hour</th>
      <th>minute</th>
    </tr>
    <tr>
      <th>ds</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-03-11 22:15:00</th>
      <td>35708.468750</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2022-03-11 22:16:00</th>
      <td>35712.550781</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2022-03-11 22:17:00</th>
      <td>35709.011719</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2022-03-11 22:18:00</th>
      <td>35722.570312</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2022-03-11 22:19:00</th>
      <td>35739.898438</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d6478d6a-0880-42a2-b433-c998d8387973')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-d6478d6a-0880-42a2-b433-c998d8387973 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d6478d6a-0880-42a2-b433-c998d8387973');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df = feature_extractor.series_lagger(df[['y']], full_data = df, n_in = 10, n_out=1, dropnan=True)
```


```python
df
```





  <div id="df-501aabd8-d14e-4a1c-929e-3e867f2cd23d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>dayofweek</th>
      <th>quarter</th>
      <th>month</th>
      <th>year</th>
      <th>dayofyear</th>
      <th>dayofmonth</th>
      <th>weekofyear</th>
      <th>hour</th>
      <th>minute</th>
      <th>lag1(t-10)</th>
      <th>lag1(t-9)</th>
      <th>lag1(t-8)</th>
      <th>lag1(t-7)</th>
      <th>lag1(t-6)</th>
      <th>lag1(t-5)</th>
      <th>lag1(t-4)</th>
      <th>lag1(t-3)</th>
      <th>lag1(t-2)</th>
      <th>lag1(t-1)</th>
    </tr>
    <tr>
      <th>ds</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-03-11 14:10:00</th>
      <td>35796.050781</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>14</td>
      <td>10</td>
      <td>35731.750000</td>
      <td>35705.480469</td>
      <td>35710.398438</td>
      <td>35730.351562</td>
      <td>35725.820312</td>
      <td>35784.769531</td>
      <td>35812.941406</td>
      <td>35820.738281</td>
      <td>35809.441406</td>
      <td>35813.289062</td>
    </tr>
    <tr>
      <th>2022-03-11 14:11:00</th>
      <td>35771.640625</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>14</td>
      <td>11</td>
      <td>35705.480469</td>
      <td>35710.398438</td>
      <td>35730.351562</td>
      <td>35725.820312</td>
      <td>35784.769531</td>
      <td>35812.941406</td>
      <td>35820.738281</td>
      <td>35809.441406</td>
      <td>35813.289062</td>
      <td>35796.050781</td>
    </tr>
    <tr>
      <th>2022-03-11 14:12:00</th>
      <td>35740.558594</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>14</td>
      <td>12</td>
      <td>35710.398438</td>
      <td>35730.351562</td>
      <td>35725.820312</td>
      <td>35784.769531</td>
      <td>35812.941406</td>
      <td>35820.738281</td>
      <td>35809.441406</td>
      <td>35813.289062</td>
      <td>35796.050781</td>
      <td>35771.640625</td>
    </tr>
    <tr>
      <th>2022-03-11 14:13:00</th>
      <td>35766.671875</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>14</td>
      <td>13</td>
      <td>35730.351562</td>
      <td>35725.820312</td>
      <td>35784.769531</td>
      <td>35812.941406</td>
      <td>35820.738281</td>
      <td>35809.441406</td>
      <td>35813.289062</td>
      <td>35796.050781</td>
      <td>35771.640625</td>
      <td>35740.558594</td>
    </tr>
    <tr>
      <th>2022-03-11 14:14:00</th>
      <td>35803.820312</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>14</td>
      <td>14</td>
      <td>35725.820312</td>
      <td>35784.769531</td>
      <td>35812.941406</td>
      <td>35820.738281</td>
      <td>35809.441406</td>
      <td>35813.289062</td>
      <td>35796.050781</td>
      <td>35771.640625</td>
      <td>35740.558594</td>
      <td>35766.671875</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-03-11 22:15:00</th>
      <td>35708.468750</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>15</td>
      <td>35705.761719</td>
      <td>35705.761719</td>
      <td>35663.738281</td>
      <td>35661.558594</td>
      <td>35686.390625</td>
      <td>35667.031250</td>
      <td>35696.550781</td>
      <td>35694.539062</td>
      <td>35707.000000</td>
      <td>35728.621094</td>
    </tr>
    <tr>
      <th>2022-03-11 22:16:00</th>
      <td>35712.550781</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>16</td>
      <td>35705.761719</td>
      <td>35663.738281</td>
      <td>35661.558594</td>
      <td>35686.390625</td>
      <td>35667.031250</td>
      <td>35696.550781</td>
      <td>35694.539062</td>
      <td>35707.000000</td>
      <td>35728.621094</td>
      <td>35708.468750</td>
    </tr>
    <tr>
      <th>2022-03-11 22:17:00</th>
      <td>35709.011719</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>17</td>
      <td>35663.738281</td>
      <td>35661.558594</td>
      <td>35686.390625</td>
      <td>35667.031250</td>
      <td>35696.550781</td>
      <td>35694.539062</td>
      <td>35707.000000</td>
      <td>35728.621094</td>
      <td>35708.468750</td>
      <td>35712.550781</td>
    </tr>
    <tr>
      <th>2022-03-11 22:18:00</th>
      <td>35722.570312</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>18</td>
      <td>35661.558594</td>
      <td>35686.390625</td>
      <td>35667.031250</td>
      <td>35696.550781</td>
      <td>35694.539062</td>
      <td>35707.000000</td>
      <td>35728.621094</td>
      <td>35708.468750</td>
      <td>35712.550781</td>
      <td>35709.011719</td>
    </tr>
    <tr>
      <th>2022-03-11 22:19:00</th>
      <td>35739.898438</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2022</td>
      <td>70</td>
      <td>11</td>
      <td>10</td>
      <td>22</td>
      <td>19</td>
      <td>35686.390625</td>
      <td>35667.031250</td>
      <td>35696.550781</td>
      <td>35694.539062</td>
      <td>35707.000000</td>
      <td>35728.621094</td>
      <td>35708.468750</td>
      <td>35712.550781</td>
      <td>35709.011719</td>
      <td>35722.570312</td>
    </tr>
  </tbody>
</table>
<p>490 rows × 20 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-501aabd8-d14e-4a1c-929e-3e867f2cd23d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-501aabd8-d14e-4a1c-929e-3e867f2cd23d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-501aabd8-d14e-4a1c-929e-3e867f2cd23d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Model Creation And Model Tuning include K-Fold Cross Validation with Library PyCaret


```python
import numpy as np
import pandas as pd
import jinja2
from pycaret.regression import *
import datetime



class statistical_models():
    def __init__(self, model_id):
      self.model_id = model_id

    def model_creator(self, df, save_path = '', target = 'y'):
        try:
            print(f'[\033[96mLOG\033[0m]Enviroment Creation Begins')
            setup(df, target = 'y', silent = True, verbose = False)
            print(f'[\033[92mSUCCESS\033[0m]Enviroment Has Created Successfully')
        except:
            print(f'[\033[91mERROR\033[0m]Enviroment Setup Error! All be demolished soon!')
        try:
            print(f'[\033[96mLOG\033[0m]{statistical_models.model_pool()[self.model_id]} Creation Begins')
            model = create_model(self.model_id, verbose = False)
            print(f'[\033[92mSUCCESS\033[0m]Model {statistical_models.model_pool()[self.model_id]} Created Successfully')
        except:
            print(f'[\033[91mERROR\033[0m]Model {statistical_models.model_pool()[self.model_id]} Not Created Successfully!')
        try:
            print(f'[\033[96mLOG\033[0m]{statistical_models.model_pool()[self.model_id]} Tuning Begins')
            tuned_model = tune_model(model, verbose = False)
            print(f'[\033[92mSUCCESS\033[0m]Model {statistical_models.model_pool()[self.model_id]} Tuning Process Has Ended Successfully')
        except:
            print(f'[\033[91mERROR\033[0m]Model {statistical_models.model_pool()[self.model_id]} Couldn\'t Tuned Successfully')

        try:
            print(f'[\033[96mLOG\033[0m]{statistical_models.model_pool()[self.model_id]} Tuned Version Saving...')
            now = datetime.datetime.now()
            save_model(tuned_model,
                      f'{save_path}{self.model_id}_model_{now.year}_{now.month}_{now.day}-{now.hour}:{now.minute}:{now.second}',
                      verbose = False)
            print(f'[\033[92mSUCCESS\033[0m]Model {statistical_models.model_pool()[self.model_id]} Saved Successfully!')
        except:
            print(f'[\033[91mERROR\033[0m]Model {statistical_models.model_pool()[self.model_id]} Couldn\'t Saved')

    def model_pool():
        models = {'lr':'Linear Regression',
                  'lasso': 'Lasso Regression',
                  'ridge': 'Ridge Regression',
                  'en': 'Elastic Net',
                  'lar': 'Least Angle Regression',
                  'llar': 'Lasso Least Angle Regression',
                  'omp': 'Orthogonal Matching Pursuit',
                  'br': 'Bayesian Ridge',
                  'ard': 'Automatic Relevance Determination',
                  'par': 'Passive Aggressive Regressor',
                  'ransac': 'Random Sample Consensus',
                  'tr': 'TheilSen Regressor',
                  'huber': 'Huber Regressor',
                  'kr': 'Kernel Ridge',
                  'svm': 'Support Vector Regression',
                  'knn': 'K Neighbours Regressor',
                  'dt': 'Decisiopn Tree Regressor',
                  'rf': 'Random Forest Regressor',
                  'et': 'Extra Trees Regressor',
                  'ada': 'AdaBoost Regressor',
                  'gbr': 'Gradient Boosting Regressor',
                  'mlp': 'MLP Regressor',
                  'xgboost': 'Extreme Gradient Boosting',
                  'lightgbm': 'Light Gradient Boosting Machine',
                  'catboost': 'CatBoost Regressor'}
        return models

```

    /usr/local/lib/python3.7/dist-packages/distributed/config.py:20: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
      defaults = yaml.load(f)



```python
for model in statistical_models.model_pool().keys():
    model_to_train = statistical_models(model)
    model_to_train.model_creator(df,
                                 save_path = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/', 
                                 target = 'y')
```

    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Linear Regression Creation Begins
    [[92mSUCCESS[0m]Model Linear Regression Created Successfully
    [[96mLOG[0m]Linear Regression Tuning Begins
    [[92mSUCCESS[0m]Model Linear Regression Tuning Process Has Ended Successfully
    [[96mLOG[0m]Linear Regression Tuned Version Saving...
    [[92mSUCCESS[0m]Model Linear Regression Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Lasso Regression Creation Begins
    [[92mSUCCESS[0m]Model Lasso Regression Created Successfully
    [[96mLOG[0m]Lasso Regression Tuning Begins
    [[92mSUCCESS[0m]Model Lasso Regression Tuning Process Has Ended Successfully
    [[96mLOG[0m]Lasso Regression Tuned Version Saving...
    [[92mSUCCESS[0m]Model Lasso Regression Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Ridge Regression Creation Begins
    [[92mSUCCESS[0m]Model Ridge Regression Created Successfully
    [[96mLOG[0m]Ridge Regression Tuning Begins
    [[92mSUCCESS[0m]Model Ridge Regression Tuning Process Has Ended Successfully
    [[96mLOG[0m]Ridge Regression Tuned Version Saving...
    [[92mSUCCESS[0m]Model Ridge Regression Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Elastic Net Creation Begins
    [[92mSUCCESS[0m]Model Elastic Net Created Successfully
    [[96mLOG[0m]Elastic Net Tuning Begins
    [[92mSUCCESS[0m]Model Elastic Net Tuning Process Has Ended Successfully
    [[96mLOG[0m]Elastic Net Tuned Version Saving...
    [[92mSUCCESS[0m]Model Elastic Net Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Least Angle Regression Creation Begins
    [[92mSUCCESS[0m]Model Least Angle Regression Created Successfully
    [[96mLOG[0m]Least Angle Regression Tuning Begins
    [[92mSUCCESS[0m]Model Least Angle Regression Tuning Process Has Ended Successfully
    [[96mLOG[0m]Least Angle Regression Tuned Version Saving...
    [[92mSUCCESS[0m]Model Least Angle Regression Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Lasso Least Angle Regression Creation Begins
    [[92mSUCCESS[0m]Model Lasso Least Angle Regression Created Successfully
    [[96mLOG[0m]Lasso Least Angle Regression Tuning Begins
    [[92mSUCCESS[0m]Model Lasso Least Angle Regression Tuning Process Has Ended Successfully
    [[96mLOG[0m]Lasso Least Angle Regression Tuned Version Saving...
    [[92mSUCCESS[0m]Model Lasso Least Angle Regression Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Orthogonal Matching Pursuit Creation Begins
    [[92mSUCCESS[0m]Model Orthogonal Matching Pursuit Created Successfully
    [[96mLOG[0m]Orthogonal Matching Pursuit Tuning Begins
    [[92mSUCCESS[0m]Model Orthogonal Matching Pursuit Tuning Process Has Ended Successfully
    [[96mLOG[0m]Orthogonal Matching Pursuit Tuned Version Saving...
    [[92mSUCCESS[0m]Model Orthogonal Matching Pursuit Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Bayesian Ridge Creation Begins
    [[92mSUCCESS[0m]Model Bayesian Ridge Created Successfully
    [[96mLOG[0m]Bayesian Ridge Tuning Begins
    [[92mSUCCESS[0m]Model Bayesian Ridge Tuning Process Has Ended Successfully
    [[96mLOG[0m]Bayesian Ridge Tuned Version Saving...
    [[92mSUCCESS[0m]Model Bayesian Ridge Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Automatic Relevance Determination Creation Begins
    [[92mSUCCESS[0m]Model Automatic Relevance Determination Created Successfully
    [[96mLOG[0m]Automatic Relevance Determination Tuning Begins
    [[92mSUCCESS[0m]Model Automatic Relevance Determination Tuning Process Has Ended Successfully
    [[96mLOG[0m]Automatic Relevance Determination Tuned Version Saving...
    [[92mSUCCESS[0m]Model Automatic Relevance Determination Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Passive Aggressive Regressor Creation Begins
    [[92mSUCCESS[0m]Model Passive Aggressive Regressor Created Successfully
    [[96mLOG[0m]Passive Aggressive Regressor Tuning Begins
    [[92mSUCCESS[0m]Model Passive Aggressive Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]Passive Aggressive Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model Passive Aggressive Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Random Sample Consensus Creation Begins
    [[92mSUCCESS[0m]Model Random Sample Consensus Created Successfully
    [[96mLOG[0m]Random Sample Consensus Tuning Begins
    [[92mSUCCESS[0m]Model Random Sample Consensus Tuning Process Has Ended Successfully
    [[96mLOG[0m]Random Sample Consensus Tuned Version Saving...
    [[92mSUCCESS[0m]Model Random Sample Consensus Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]TheilSen Regressor Creation Begins
    [[92mSUCCESS[0m]Model TheilSen Regressor Created Successfully
    [[96mLOG[0m]TheilSen Regressor Tuning Begins
    [[92mSUCCESS[0m]Model TheilSen Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]TheilSen Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model TheilSen Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Huber Regressor Creation Begins
    [[92mSUCCESS[0m]Model Huber Regressor Created Successfully
    [[96mLOG[0m]Huber Regressor Tuning Begins
    [[92mSUCCESS[0m]Model Huber Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]Huber Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model Huber Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Kernel Ridge Creation Begins
    [[92mSUCCESS[0m]Model Kernel Ridge Created Successfully
    [[96mLOG[0m]Kernel Ridge Tuning Begins
    [[92mSUCCESS[0m]Model Kernel Ridge Tuning Process Has Ended Successfully
    [[96mLOG[0m]Kernel Ridge Tuned Version Saving...
    [[92mSUCCESS[0m]Model Kernel Ridge Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Support Vector Regression Creation Begins
    [[92mSUCCESS[0m]Model Support Vector Regression Created Successfully
    [[96mLOG[0m]Support Vector Regression Tuning Begins
    [[92mSUCCESS[0m]Model Support Vector Regression Tuning Process Has Ended Successfully
    [[96mLOG[0m]Support Vector Regression Tuned Version Saving...
    [[92mSUCCESS[0m]Model Support Vector Regression Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]K Neighbours Regressor Creation Begins
    [[92mSUCCESS[0m]Model K Neighbours Regressor Created Successfully
    [[96mLOG[0m]K Neighbours Regressor Tuning Begins
    [[92mSUCCESS[0m]Model K Neighbours Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]K Neighbours Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model K Neighbours Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Decisiopn Tree Regressor Creation Begins
    [[92mSUCCESS[0m]Model Decisiopn Tree Regressor Created Successfully
    [[96mLOG[0m]Decisiopn Tree Regressor Tuning Begins
    [[92mSUCCESS[0m]Model Decisiopn Tree Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]Decisiopn Tree Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model Decisiopn Tree Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Random Forest Regressor Creation Begins
    [[92mSUCCESS[0m]Model Random Forest Regressor Created Successfully
    [[96mLOG[0m]Random Forest Regressor Tuning Begins
    [[92mSUCCESS[0m]Model Random Forest Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]Random Forest Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model Random Forest Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Extra Trees Regressor Creation Begins
    [[92mSUCCESS[0m]Model Extra Trees Regressor Created Successfully
    [[96mLOG[0m]Extra Trees Regressor Tuning Begins
    [[92mSUCCESS[0m]Model Extra Trees Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]Extra Trees Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model Extra Trees Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]AdaBoost Regressor Creation Begins
    [[92mSUCCESS[0m]Model AdaBoost Regressor Created Successfully
    [[96mLOG[0m]AdaBoost Regressor Tuning Begins
    [[92mSUCCESS[0m]Model AdaBoost Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]AdaBoost Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model AdaBoost Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Gradient Boosting Regressor Creation Begins
    [[92mSUCCESS[0m]Model Gradient Boosting Regressor Created Successfully
    [[96mLOG[0m]Gradient Boosting Regressor Tuning Begins
    [[92mSUCCESS[0m]Model Gradient Boosting Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]Gradient Boosting Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model Gradient Boosting Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]MLP Regressor Creation Begins
    [[92mSUCCESS[0m]Model MLP Regressor Created Successfully
    [[96mLOG[0m]MLP Regressor Tuning Begins
    [[92mSUCCESS[0m]Model MLP Regressor Tuning Process Has Ended Successfully
    [[96mLOG[0m]MLP Regressor Tuned Version Saving...
    [[92mSUCCESS[0m]Model MLP Regressor Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Extreme Gradient Boosting Creation Begins
    [[91mERROR[0m]Model Extreme Gradient Boosting Not Created Successfully!
    [[96mLOG[0m]Extreme Gradient Boosting Tuning Begins
    [[91mERROR[0m]Model Extreme Gradient Boosting Couldn't Tuned Successfully
    [[96mLOG[0m]Extreme Gradient Boosting Tuned Version Saving...
    [[91mERROR[0m]Model Extreme Gradient Boosting Couldn't Saved
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]Light Gradient Boosting Machine Creation Begins
    [[92mSUCCESS[0m]Model Light Gradient Boosting Machine Created Successfully
    [[96mLOG[0m]Light Gradient Boosting Machine Tuning Begins
    [[92mSUCCESS[0m]Model Light Gradient Boosting Machine Tuning Process Has Ended Successfully
    [[96mLOG[0m]Light Gradient Boosting Machine Tuned Version Saving...
    [[92mSUCCESS[0m]Model Light Gradient Boosting Machine Saved Successfully!
    [[96mLOG[0m]Enviroment Creation Begins
    [[92mSUCCESS[0m]Enviroment Has Created Successfully
    [[96mLOG[0m]CatBoost Regressor Creation Begins
    [[91mERROR[0m]Model CatBoost Regressor Not Created Successfully!
    [[96mLOG[0m]CatBoost Regressor Tuning Begins
    [[91mERROR[0m]Model CatBoost Regressor Couldn't Tuned Successfully
    [[96mLOG[0m]CatBoost Regressor Tuned Version Saving...
    [[91mERROR[0m]Model CatBoost Regressor Couldn't Saved


# Forecastor Module Include Argument Parsing(Disabled for Jupyter)


```python
from pycaret.regression import load_model, predict_model
#from preprocessor import preprocessing
#from feature_extractor import feature_extractor
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
                pred = predict_model(model, last_df, verbose = False)['Label'].values[0]
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

'''
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
'''

```


```python
forecastor(df[:-10],
           model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl',
           forecast_range_min = 110,
           num_of_lag = 10)
```

    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created






  <div id="df-498cb73c-8860-4d53-91f7-9a49e276a174">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dates</th>
      <th>Forecasts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-03-11 22:10:00</td>
      <td>35676.83146686491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-03-11 22:11:00</td>
      <td>35689.66466094872</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-03-11 22:12:00</td>
      <td>35695.569369075834</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-03-11 22:13:00</td>
      <td>35703.68712122983</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-03-11 22:14:00</td>
      <td>35709.00290609244</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2022-03-11 23:55:00</td>
      <td>35750.32621231268</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2022-03-11 23:56:00</td>
      <td>35749.68377529154</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2022-03-11 23:57:00</td>
      <td>35749.03798695729</td>
    </tr>
    <tr>
      <th>108</th>
      <td>2022-03-11 23:58:00</td>
      <td>35748.39165365355</td>
    </tr>
    <tr>
      <th>109</th>
      <td>2022-03-11 23:59:00</td>
      <td>35747.74230523598</td>
    </tr>
  </tbody>
</table>
<p>110 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-498cb73c-8860-4d53-91f7-9a49e276a174')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-498cb73c-8860-4d53-91f7-9a49e276a174 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-498cb73c-8860-4d53-91f7-9a49e276a174');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
real = df[20:400].y.values


forecasts = forecastor(df[:20],
                        model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl',
                        forecast_range_min = 380,
                        num_of_lag = 10).Forecasts.values
yhat = [float(i) for i in forecasts]
```

    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created


# Error Metrics


```python
import numpy as np

class error_metrics:
    def mape(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def mse(y_true, y_pred):
        return np.square(np.subtract(y_true, y_pred)).mean()

    def mae(y_true, y_pred):
        return np.sum(np.absolute((y_true - y_pred)))

    def rmse(y_true, y_pred):
        rms = np.sqrt(error_metrics.mse(y_true, y_pred))
        return rms
```

# Analyzing All Models and Selecting Better Models Between Them


```python
import matplotlib.pyplot as plt
import glob
import re

paths = glob.glob('/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/*.pkl',recursive = True)
files = os.listdir(r'/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/')

real = df[20:400].y.values

for i in range(len(paths)):
    forecasts = forecastor(df[:20],
                            model = paths[i],
                            forecast_range_min = 380,
                            num_of_lag = 10).Forecasts.values
    yhat = [float(i) for i in forecasts]

    plt.style.use(['dark_background'])
    plt.figure(figsize=(20,10))
    plt.plot(real, 'm')
    plt.plot(yhat , 'c')
    plt.title(files[i])
    plt.ylabel('$', fontsize=20)
    plt.xlabel('MAPE:%.2f / MSE: %.2f / MAE: %.2f / RMSE: %.2f' % (error_metrics.mape(real,yhat),
                                                                  error_metrics.mse(real,yhat),
                                                                  error_metrics.mae(real,yhat),
                                                                  error_metrics.rmse(real,yhat)),
              fontsize=20)
    plt.grid(color='y', linestyle='--', linewidth=0.5)
    plt.show()

```

    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lr_model_2022_6_26-11:24:45.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_1.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lasso_model_2022_6_26-11:24:48.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_3.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ridge_model_2022_6_26-11:24:51.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_5.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/en_model_2022_6_26-11:24:54.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_7.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lar_model_2022_6_26-11:24:57.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_9.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/llar_model_2022_6_26-11:24:59.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_11.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_13.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/br_model_2022_6_26-11:25:4.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_15.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ard_model_2022_6_26-11:25:14.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_17.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/par_model_2022_6_26-11:25:18.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_19.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ransac_model_2022_6_26-11:25:21.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_21.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/tr_model_2022_6_26-11:26:54.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_23.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/huber_model_2022_6_26-11:27:1.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_25.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/kr_model_2022_6_26-11:27:4.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_27.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/svm_model_2022_6_26-11:27:8.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_29.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/knn_model_2022_6_26-11:27:16.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_31.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/dt_model_2022_6_26-11:27:20.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_33.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/rf_model_2022_6_26-11:28:54.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_35.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/et_model_2022_6_26-11:30:8.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_37.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ada_model_2022_6_26-11:30:46.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_39.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/gbr_model_2022_6_26-11:31:4.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_41.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/mlp_model_2022_6_26-11:32:49.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_43.png)
    


    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lightgbm_model_2022_6_26-11:32:58.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created



    
![png](output_20_45.png)
    


# Better Models


```python
better_models = {'lr':'Linear Regression',
                  'lasso': 'Lasso Regression',
                  'ridge': 'Ridge Regression',
                  'lar': 'Least Angle Regression',
                  'omp': 'Orthogonal Matching Pursuit',
                  'br': 'Bayesian Ridge',
                  'ransac': 'Random Sample Consensus',
                  'tr': 'TheilSen Regressor',
                  'lightgbm': 'Light Gradient Boosting Machine'
                  }
```


```python
lr = forecastor(df[:20],
                model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lr_model_2022_6_26-11:24:45.pkl',
                forecast_range_min = 380,
                num_of_lag = 10).Forecasts.values

lasso = forecastor(df[:20],
                   model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lasso_model_2022_6_26-11:24:48.pkl',
                   forecast_range_min = 380,
                   num_of_lag = 10).Forecasts.values

ridge = forecastor(df[:20],
                   model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ridge_model_2022_6_26-11:24:51.pkl',
                   forecast_range_min = 380,
                   num_of_lag = 10).Forecasts.values

lar = lasso = forecastor(df[:20],
                         model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lar_model_2022_6_26-11:24:57.pkl',
                         forecast_range_min = 380,
                         num_of_lag = 10).Forecasts.values

omp = forecastor(df[:20],
                 model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl',
                 forecast_range_min = 380,
                 num_of_lag = 10).Forecasts.values

br = forecastor(df[:20],
                model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/br_model_2022_6_26-11:25:4.pkl',
                forecast_range_min = 380,
                num_of_lag = 10).Forecasts.values

ransac = forecastor(df[:20],
                    model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ransac_model_2022_6_26-11:25:21.pkl',
                    forecast_range_min = 380,
                    num_of_lag = 10).Forecasts.values

tr = forecastor(df[:20],
                model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/tr_model_2022_6_26-11:26:54.pkl',
                forecast_range_min = 380,
                num_of_lag = 10).Forecasts.values

lightgbm = forecastor(df[:20],
                      model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lightgbm_model_2022_6_26-11:32:58.pkl',
                      forecast_range_min = 380,
                      num_of_lag = 10).Forecasts.values

```

    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lr_model_2022_6_26-11:24:45.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lasso_model_2022_6_26-11:24:48.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ridge_model_2022_6_26-11:24:51.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lar_model_2022_6_26-11:24:57.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/br_model_2022_6_26-11:25:4.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ransac_model_2022_6_26-11:25:21.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/tr_model_2022_6_26-11:26:54.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lightgbm_model_2022_6_26-11:32:58.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created


# Better Forecast Analyzing with Selected Models


```python
better_forecasts = [lr,
                    lasso,
                    ridge,
                    lar,
                    omp,
                    br,
                    ransac,
                    tr,
                    lightgbm]
```


```python
excellent_forecasts = []

for i in range(len(better_forecasts[0])):
    excellent_forecasts.append(np.mean([float(j[i]) for j in better_forecasts]))
```


```python
yhat = excellent_forecasts.copy()

plt.style.use(['dark_background'])
plt.figure(figsize=(20,10))
plt.plot(real, 'm')
plt.plot(yhat , 'c')
plt.title('Perfection')
plt.ylabel('$', fontsize=20)
plt.xlabel('MAPE:%.2f / MSE: %.2f / MAE: %.2f / RMSE: %.2f' % (error_metrics.mape(real,yhat),
                                                              error_metrics.mse(real,yhat),
                                                              error_metrics.mae(real,yhat),
                                                              error_metrics.rmse(real,yhat)),
          fontsize=20)
plt.grid(color='y', linestyle='--', linewidth=0.5)
plt.show()
```


    
![png](output_27_0.png)
    



```python
better_forecasts_paths = ['/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lr_model_2022_6_26-11:24:45.pkl',
                          '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lasso_model_2022_6_26-11:24:48.pkl',
                          '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ridge_model_2022_6_26-11:24:51.pkl',
                          '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lar_model_2022_6_26-11:24:57.pkl',
                          '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl',
                          '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/br_model_2022_6_26-11:25:4.pkl',
                          '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ransac_model_2022_6_26-11:25:21.pkl',
                          '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/tr_model_2022_6_26-11:26:54.pkl',
                          '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lightgbm_model_2022_6_26-11:32:58.pkl']
```


```python
# Test Forecast
future = [
          forecastor(df[:400],
                      model = path,
                      forecast_range_min = 150,
                      num_of_lag = 10).Forecasts.values for path in better_forecasts_paths
         ]

```

    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lr_model_2022_6_26-11:24:45.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lasso_model_2022_6_26-11:24:48.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ridge_model_2022_6_26-11:24:51.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lar_model_2022_6_26-11:24:57.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/br_model_2022_6_26-11:25:4.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ransac_model_2022_6_26-11:25:21.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/tr_model_2022_6_26-11:26:54.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created
    [[96mLOG[0m]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lightgbm_model_2022_6_26-11:32:58.pkl
    Transformation Pipeline and Model Successfully Loaded
    [[92mSUCCESS[0m]Model Loaded Successfully
    [[96mLOG[0m]Obtaining Data
    [[92mSUCCESS[0m]Data Successfully Obtained
    [[96mLOG[0m]Forecasting Begining
    [[92mSUCCESS[0m]Forecasting Successfully Done
    [[96mLOG[0m]Creating Forecast Dataframe
    [[92mSUCCESS[0m]Forecasting Dataframe Successfully Created


# Forecasting Future of BTC - EUR Compharison


```python
excellent_future = []
for i in range(150):
    excellent_future.append(np.mean([float(j[i]) for j in future]))
```


```python
full_yhat = np.append(yhat, excellent_future)

plt.style.use(['dark_background'])
plt.figure(figsize=(20,10))
plt.plot(real, 'm')
plt.plot(full_yhat , 'c')
plt.title('Future 150 Minutes Forecast', fontsize=20)
plt.ylabel('$', fontsize=20)
plt.xlabel('MAPE:%.2f / MSE: %.2f / MAE: %.2f / RMSE: %.2f' % (error_metrics.mape(real,yhat),
                                                              error_metrics.mse(real,yhat),
                                                              error_metrics.mae(real,yhat),
                                                              error_metrics.rmse(real,yhat)),
          fontsize=20)
plt.grid(color='y', linestyle='--', linewidth=0.5)
plt.show()
```


    
![png](output_32_0.png)
    


# Thank You :D
