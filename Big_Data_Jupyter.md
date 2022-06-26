::: {#MathJax_Message style="display: none;"}
:::

::: {#notebook .border-box-sizing tabindex="-1"}
::: {#notebook-container .container}
::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Preprocessing to for All Bionance Historical Dataset[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Preprocessing-to-for-All-Bionance-Historical-Dataset){.anchor-link} {#Preprocessing-to-for-All-Binance-Historical-Dataset}
================================================================================================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[7\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[8\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
df = preprocessing.df_pp('/content/drive/MyDrive/BIG_DATA_PROJECT/Dataset/BTC-EUR.parquet', 'T')
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
``` {style="position: relative;"}
[LOG]Data /content/drive/MyDrive/BIG_DATA_PROJECT/Dataset/BTC-EUR.parquet Reading
[SUCCESS]Data Read Successfully
[LOG]Data Conversion to /content/drive/MyDrive/BIG_DATA_PROJECT/Dataset/BTC-EUR.parquet Begins.
[SUCCESS]Data Converted Successfully
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[9\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
df.tail()
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt .output_prompt}
Out\[9\]:
:::

::: {.output_html .rendered_html .output_subarea .output_execute_result}
::: {#df-741d5971-bc60-42f0-a990-0e98f9b13d7d}
::: {.colab-df-container}
<div>

        ds                    y
  ----- --------------------- --------------
  495   2022-03-11 22:15:00   35708.468750
  496   2022-03-11 22:16:00   35712.550781
  497   2022-03-11 22:17:00   35709.011719
  498   2022-03-11 22:18:00   35722.570312
  499   2022-03-11 22:19:00   35739.898438

</div>
:::
:::
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Extracting Future form Dataset and Historical Values as Lag for Forecasting[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Extracting-Future-form-Dataset-and-Historical-Values-as-Lag-for-Forecasting){.anchor-link} {#Extracting-Future-form-Dataset-and-Historical-Values-as-Lag-for-Forecasting}
==============================================================================================================================================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[10\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[11\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
df = feature_extractor.create_features(df)
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stderr .output_text}
``` {style="position: relative;"}
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:23: FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.  Please use Series.dt.isocalendar().week instead.
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[12\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
df.tail()
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt .output_prompt}
Out\[12\]:
:::

::: {.output_html .rendered_html .output_subarea .output_execute_result}
::: {#df-d6478d6a-0880-42a2-b433-c998d8387973}
::: {.colab-df-container}
<div>

</div>
:::
:::
:::
:::
:::
:::
:::
:::
:::

y

dayofweek

quarter

month

year

dayofyear

dayofmonth

weekofyear

hour

minute

ds

2022-03-11 22:15:00

35708.468750

4

1

3

2022

70

11

10

22

15

2022-03-11 22:16:00

35712.550781

4

1

3

2022

70

11

10

22

16

2022-03-11 22:17:00

35709.011719

4

1

3

2022

70

11

10

22

17

2022-03-11 22:18:00

35722.570312

4

1

3

2022

70

11

10

22

18

2022-03-11 22:19:00

35739.898438

4

1

3

2022

70

11

10

22

19

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[13\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
df = feature_extractor.series_lagger(df[['y']], full_data = df, n_in = 10, n_out=1, dropnan=True)
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[14\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
df
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt .output_prompt}
Out\[14\]:
:::

::: {.output_html .rendered_html .output_subarea .output_execute_result}
::: {#df-501aabd8-d14e-4a1c-929e-3e867f2cd23d}
::: {.colab-df-container}
<div>

</div>
:::
:::
:::
:::
:::
:::
:::

y

dayofweek

quarter

month

year

dayofyear

dayofmonth

weekofyear

hour

minute

lag1(t-10)

lag1(t-9)

lag1(t-8)

lag1(t-7)

lag1(t-6)

lag1(t-5)

lag1(t-4)

lag1(t-3)

lag1(t-2)

lag1(t-1)

ds

2022-03-11 14:10:00

35796.050781

4

1

3

2022

70

11

10

14

10

35731.750000

35705.480469

35710.398438

35730.351562

35725.820312

35784.769531

35812.941406

35820.738281

35809.441406

35813.289062

2022-03-11 14:11:00

35771.640625

4

1

3

2022

70

11

10

14

11

35705.480469

35710.398438

35730.351562

35725.820312

35784.769531

35812.941406

35820.738281

35809.441406

35813.289062

35796.050781

2022-03-11 14:12:00

35740.558594

4

1

3

2022

70

11

10

14

12

35710.398438

35730.351562

35725.820312

35784.769531

35812.941406

35820.738281

35809.441406

35813.289062

35796.050781

35771.640625

2022-03-11 14:13:00

35766.671875

4

1

3

2022

70

11

10

14

13

35730.351562

35725.820312

35784.769531

35812.941406

35820.738281

35809.441406

35813.289062

35796.050781

35771.640625

35740.558594

2022-03-11 14:14:00

35803.820312

4

1

3

2022

70

11

10

14

14

35725.820312

35784.769531

35812.941406

35820.738281

35809.441406

35813.289062

35796.050781

35771.640625

35740.558594

35766.671875

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

\...

2022-03-11 22:15:00

35708.468750

4

1

3

2022

70

11

10

22

15

35705.761719

35705.761719

35663.738281

35661.558594

35686.390625

35667.031250

35696.550781

35694.539062

35707.000000

35728.621094

2022-03-11 22:16:00

35712.550781

4

1

3

2022

70

11

10

22

16

35705.761719

35663.738281

35661.558594

35686.390625

35667.031250

35696.550781

35694.539062

35707.000000

35728.621094

35708.468750

2022-03-11 22:17:00

35709.011719

4

1

3

2022

70

11

10

22

17

35663.738281

35661.558594

35686.390625

35667.031250

35696.550781

35694.539062

35707.000000

35728.621094

35708.468750

35712.550781

2022-03-11 22:18:00

35722.570312

4

1

3

2022

70

11

10

22

18

35661.558594

35686.390625

35667.031250

35696.550781

35694.539062

35707.000000

35728.621094

35708.468750

35712.550781

35709.011719

2022-03-11 22:19:00

35739.898438

4

1

3

2022

70

11

10

22

19

35686.390625

35667.031250

35696.550781

35694.539062

35707.000000

35728.621094

35708.468750

35712.550781

35709.011719

35722.570312

490 rows × 20 columns

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Model Creation And Model Tuning include K-Fold Cross Validation with Library PyCaret[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Model-Creation-And-Model-Tuning-include-K-Fold-Cross-Validation-with-Library-PyCaret){.anchor-link} {#Model-Creation-And-Model-Tuning-include-K-Fold-Cross-Validation-with-Library-PyCaret}
================================================================================================================================================================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[15\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stderr .output_text}
``` {style="position: relative;"}
/usr/local/lib/python3.7/dist-packages/distributed/config.py:20: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  defaults = yaml.load(f)
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[16\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
for model in statistical_models.model_pool().keys():
    model_to_train = statistical_models(model)
    model_to_train.model_creator(df,
                                 save_path = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/', 
                                 target = 'y')
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
``` {style="position: relative;"}
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Linear Regression Creation Begins
[SUCCESS]Model Linear Regression Created Successfully
[LOG]Linear Regression Tuning Begins
[SUCCESS]Model Linear Regression Tuning Process Has Ended Successfully
[LOG]Linear Regression Tuned Version Saving...
[SUCCESS]Model Linear Regression Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Lasso Regression Creation Begins
[SUCCESS]Model Lasso Regression Created Successfully
[LOG]Lasso Regression Tuning Begins
[SUCCESS]Model Lasso Regression Tuning Process Has Ended Successfully
[LOG]Lasso Regression Tuned Version Saving...
[SUCCESS]Model Lasso Regression Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Ridge Regression Creation Begins
[SUCCESS]Model Ridge Regression Created Successfully
[LOG]Ridge Regression Tuning Begins
[SUCCESS]Model Ridge Regression Tuning Process Has Ended Successfully
[LOG]Ridge Regression Tuned Version Saving...
[SUCCESS]Model Ridge Regression Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Elastic Net Creation Begins
[SUCCESS]Model Elastic Net Created Successfully
[LOG]Elastic Net Tuning Begins
[SUCCESS]Model Elastic Net Tuning Process Has Ended Successfully
[LOG]Elastic Net Tuned Version Saving...
[SUCCESS]Model Elastic Net Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Least Angle Regression Creation Begins
[SUCCESS]Model Least Angle Regression Created Successfully
[LOG]Least Angle Regression Tuning Begins
[SUCCESS]Model Least Angle Regression Tuning Process Has Ended Successfully
[LOG]Least Angle Regression Tuned Version Saving...
[SUCCESS]Model Least Angle Regression Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Lasso Least Angle Regression Creation Begins
[SUCCESS]Model Lasso Least Angle Regression Created Successfully
[LOG]Lasso Least Angle Regression Tuning Begins
[SUCCESS]Model Lasso Least Angle Regression Tuning Process Has Ended Successfully
[LOG]Lasso Least Angle Regression Tuned Version Saving...
[SUCCESS]Model Lasso Least Angle Regression Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Orthogonal Matching Pursuit Creation Begins
[SUCCESS]Model Orthogonal Matching Pursuit Created Successfully
[LOG]Orthogonal Matching Pursuit Tuning Begins
[SUCCESS]Model Orthogonal Matching Pursuit Tuning Process Has Ended Successfully
[LOG]Orthogonal Matching Pursuit Tuned Version Saving...
[SUCCESS]Model Orthogonal Matching Pursuit Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Bayesian Ridge Creation Begins
[SUCCESS]Model Bayesian Ridge Created Successfully
[LOG]Bayesian Ridge Tuning Begins
[SUCCESS]Model Bayesian Ridge Tuning Process Has Ended Successfully
[LOG]Bayesian Ridge Tuned Version Saving...
[SUCCESS]Model Bayesian Ridge Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Automatic Relevance Determination Creation Begins
[SUCCESS]Model Automatic Relevance Determination Created Successfully
[LOG]Automatic Relevance Determination Tuning Begins
[SUCCESS]Model Automatic Relevance Determination Tuning Process Has Ended Successfully
[LOG]Automatic Relevance Determination Tuned Version Saving...
[SUCCESS]Model Automatic Relevance Determination Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Passive Aggressive Regressor Creation Begins
[SUCCESS]Model Passive Aggressive Regressor Created Successfully
[LOG]Passive Aggressive Regressor Tuning Begins
[SUCCESS]Model Passive Aggressive Regressor Tuning Process Has Ended Successfully
[LOG]Passive Aggressive Regressor Tuned Version Saving...
[SUCCESS]Model Passive Aggressive Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Random Sample Consensus Creation Begins
[SUCCESS]Model Random Sample Consensus Created Successfully
[LOG]Random Sample Consensus Tuning Begins
[SUCCESS]Model Random Sample Consensus Tuning Process Has Ended Successfully
[LOG]Random Sample Consensus Tuned Version Saving...
[SUCCESS]Model Random Sample Consensus Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]TheilSen Regressor Creation Begins
[SUCCESS]Model TheilSen Regressor Created Successfully
[LOG]TheilSen Regressor Tuning Begins
[SUCCESS]Model TheilSen Regressor Tuning Process Has Ended Successfully
[LOG]TheilSen Regressor Tuned Version Saving...
[SUCCESS]Model TheilSen Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Huber Regressor Creation Begins
[SUCCESS]Model Huber Regressor Created Successfully
[LOG]Huber Regressor Tuning Begins
[SUCCESS]Model Huber Regressor Tuning Process Has Ended Successfully
[LOG]Huber Regressor Tuned Version Saving...
[SUCCESS]Model Huber Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Kernel Ridge Creation Begins
[SUCCESS]Model Kernel Ridge Created Successfully
[LOG]Kernel Ridge Tuning Begins
[SUCCESS]Model Kernel Ridge Tuning Process Has Ended Successfully
[LOG]Kernel Ridge Tuned Version Saving...
[SUCCESS]Model Kernel Ridge Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Support Vector Regression Creation Begins
[SUCCESS]Model Support Vector Regression Created Successfully
[LOG]Support Vector Regression Tuning Begins
[SUCCESS]Model Support Vector Regression Tuning Process Has Ended Successfully
[LOG]Support Vector Regression Tuned Version Saving...
[SUCCESS]Model Support Vector Regression Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]K Neighbours Regressor Creation Begins
[SUCCESS]Model K Neighbours Regressor Created Successfully
[LOG]K Neighbours Regressor Tuning Begins
[SUCCESS]Model K Neighbours Regressor Tuning Process Has Ended Successfully
[LOG]K Neighbours Regressor Tuned Version Saving...
[SUCCESS]Model K Neighbours Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Decisiopn Tree Regressor Creation Begins
[SUCCESS]Model Decisiopn Tree Regressor Created Successfully
[LOG]Decisiopn Tree Regressor Tuning Begins
[SUCCESS]Model Decisiopn Tree Regressor Tuning Process Has Ended Successfully
[LOG]Decisiopn Tree Regressor Tuned Version Saving...
[SUCCESS]Model Decisiopn Tree Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Random Forest Regressor Creation Begins
[SUCCESS]Model Random Forest Regressor Created Successfully
[LOG]Random Forest Regressor Tuning Begins
[SUCCESS]Model Random Forest Regressor Tuning Process Has Ended Successfully
[LOG]Random Forest Regressor Tuned Version Saving...
[SUCCESS]Model Random Forest Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Extra Trees Regressor Creation Begins
[SUCCESS]Model Extra Trees Regressor Created Successfully
[LOG]Extra Trees Regressor Tuning Begins
[SUCCESS]Model Extra Trees Regressor Tuning Process Has Ended Successfully
[LOG]Extra Trees Regressor Tuned Version Saving...
[SUCCESS]Model Extra Trees Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]AdaBoost Regressor Creation Begins
[SUCCESS]Model AdaBoost Regressor Created Successfully
[LOG]AdaBoost Regressor Tuning Begins
[SUCCESS]Model AdaBoost Regressor Tuning Process Has Ended Successfully
[LOG]AdaBoost Regressor Tuned Version Saving...
[SUCCESS]Model AdaBoost Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Gradient Boosting Regressor Creation Begins
[SUCCESS]Model Gradient Boosting Regressor Created Successfully
[LOG]Gradient Boosting Regressor Tuning Begins
[SUCCESS]Model Gradient Boosting Regressor Tuning Process Has Ended Successfully
[LOG]Gradient Boosting Regressor Tuned Version Saving...
[SUCCESS]Model Gradient Boosting Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]MLP Regressor Creation Begins
[SUCCESS]Model MLP Regressor Created Successfully
[LOG]MLP Regressor Tuning Begins
[SUCCESS]Model MLP Regressor Tuning Process Has Ended Successfully
[LOG]MLP Regressor Tuned Version Saving...
[SUCCESS]Model MLP Regressor Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Extreme Gradient Boosting Creation Begins
[ERROR]Model Extreme Gradient Boosting Not Created Successfully!
[LOG]Extreme Gradient Boosting Tuning Begins
[ERROR]Model Extreme Gradient Boosting Couldn't Tuned Successfully
[LOG]Extreme Gradient Boosting Tuned Version Saving...
[ERROR]Model Extreme Gradient Boosting Couldn't Saved
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]Light Gradient Boosting Machine Creation Begins
[SUCCESS]Model Light Gradient Boosting Machine Created Successfully
[LOG]Light Gradient Boosting Machine Tuning Begins
[SUCCESS]Model Light Gradient Boosting Machine Tuning Process Has Ended Successfully
[LOG]Light Gradient Boosting Machine Tuned Version Saving...
[SUCCESS]Model Light Gradient Boosting Machine Saved Successfully!
[LOG]Enviroment Creation Begins
[SUCCESS]Enviroment Has Created Successfully
[LOG]CatBoost Regressor Creation Begins
[ERROR]Model CatBoost Regressor Not Created Successfully!
[LOG]CatBoost Regressor Tuning Begins
[ERROR]Model CatBoost Regressor Couldn't Tuned Successfully
[LOG]CatBoost Regressor Tuned Version Saving...
[ERROR]Model CatBoost Regressor Couldn't Saved
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Forecastor Module Include Argument Parsing(Disabled for Jupyter)[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Forecastor-Module-Include-Argument-Parsing(Disabled-for-Jupyter)){.anchor-link} {#Forecastor-Module-Include-Argument-Parsing(Disabled-for-Jupyter)}
========================================================================================================================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[ \]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[29\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
forecastor(df[:-10],
           model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl',
           forecast_range_min = 110,
           num_of_lag = 10)
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
``` {style="position: relative;"}
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[36\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
real = df[20:400].y.values


forecasts = forecastor(df[:20],
                        model = '/content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl',
                        forecast_range_min = 380,
                        num_of_lag = 10).Forecasts.values
yhat = [float(i) for i in forecasts]
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
``` {style="position: relative;"}
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Error Metrics[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Error-Metrics){.anchor-link} {#Error-Metrics}
==================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[60\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Analyzing All Models and Selecting Better Models Between Them[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Analyzing-All-Models-and-Selecting-Better-Models-Between-Them){.anchor-link} {#Analyzing-All-Models-and-Selecting-Better-Models-Between-Them}
==================================================================================================================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[73\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
``` {style="position: relative;"}
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lr_model_2022_6_26-11:24:45.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Better Models[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Better-Models){.anchor-link} {#Better-Models}
==================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[76\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[77\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
``` {style="position: relative;"}
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lr_model_2022_6_26-11:24:45.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lasso_model_2022_6_26-11:24:48.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ridge_model_2022_6_26-11:24:51.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lar_model_2022_6_26-11:24:57.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/br_model_2022_6_26-11:25:4.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ransac_model_2022_6_26-11:25:21.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/tr_model_2022_6_26-11:26:54.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lightgbm_model_2022_6_26-11:32:58.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Better Forecast Analyzing with Selected Models[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Better-Forecast-Analyzing-with-Selected-Models){.anchor-link} {#Better-Forecast-Analyzing-with-Selected-Models}
====================================================================================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[78\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[88\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
excellent_forecasts = []

for i in range(len(better_forecasts[0])):
    excellent_forecasts.append(np.mean([float(j[i]) for j in better_forecasts]))
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[91\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_png .output_subarea}
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABKQAAAJkCAYAAAAvG/leAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdd5xV9Z3/8de5U5gGM3SQjooaLOgKFoogSlTsBUWNiq4mrr/NkpisGk1iNiZmNRqi2URNjDERE7CFWGKLoIAKVmyIooLS6zBMb9/fH7cMc+ZzYGa4hTm8n4/HfTzcO3PvnOu8JOvH7/d7PMAhIiIiIiIiIiKSJpFMX4CIiIiIiIiIiOxdNJASEREREREREZG00kBKRERERERERETSSgMpERERERERERFJKw2kREREREREREQkrTSQEhERERERERGRtNJASkRERCQJnHN8+umnLF26lGXLlrF48WKOP/74Nr9Ply5deOedd/jkk0/o1q1bm1+fm5vLN77xDQD22Wcf3n///Ta/h4iIiEiqeYDL9EWIiIiIdHTOOfr378/q1asBOPbYY3nyySc54IAD2LRpU6vfZ8yYMTz88MMMHDiwXddx1FFHccstt3DiiSe26/UiIiIi6aAVUiIiIiIp8Oqrr7J8+XKOOeYYAE4//XTee+89PvvsM5577jm6d+8OwI9//GPuu+8+Fi1axA9/+ENmzpxJ7969Wbp0Kd27d+fYY49l8eLFfPrpp7z22msMGTIk8TPuuOMOPv/8c5YtW8b3vvc9evXqxRNPPMExxxzDK6+8wqBBg6irqwPA8zxuueUWli5dytKlS3nggQcoKCgAYO7cuXznO99h/vz5rFq1iocffjjNf7dERERkb+T00EMPPfTQQw899Ni9h3PO9evXr9lzb7/9tps0aZIbMmSI27Ztmxs+fLgD3PXXX+8eeeQRB7gf//jHbtWqVa579+4OcMcdd5z79NNPHeCKiorc5s2b3QknnOAAd8EFF7g33njDAe6iiy5y8+fPd9nZ2a5z585u5cqVbuTIke7SSy91L7zwggPcoEGDXF1dnQPc+eef79566y1XUFDgIpGIe+KJJ9yNN97oADd37lz30ksvuby8PFdQUODWrVvnjj322Iz/PdVDDz300EMPPcL70AopERERkRQ46aST6NOnDwsX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xBmRWTIzM5GWlgYA+OGHH3D8+HG0bNkS4eHhAICkpKQ6l+qQfRUUFPhlwff6wYw4z2effYbt27cDAD744ANcc801zIUD3XzzzXjqqacwcuRInD17ltcL8ssErxXUt29ftGvXDgCwfft2hISEID8/n9cKh9NysXPnTl4vqE4sSNnEypUrceuttwIA+vTpg8zMTBQUFJg8KzLDlClT8MgjjwAAWrdujdatW+Nf//oXJk6cCACYOHEiVqxYYeYUyUSrVq3yy8KmTZtw3XXXITY2FlFRUUhJScG6detMnik1pffffx+dOnUCULnP2K5du5gLh4mJicHcuXPx4x//GKdPnwbA64XTaZngtYJSU1M9v2cmJiaiefPmvFaQmot58+bxekH1YvrO6hwXZvzxj3+UjIwMWbdunfTs2dP0+XCYM5o3by4ffPCBfP755/LFF1/IyJEjpU2bNrJy5Ur5/PPPZcGCBRISEmL6PDku/ujbt6+kp6fLgQMHZN++fZKeni6XXXaZmoWJEyfKF198IRs3bpQpU6aYPneOps3FLbfcIps3b5Y1a9bIhx9+KAkJCcyFw8bMmTPl2LFjkp6e7hkdOnTg9cLBQ8vE1KlTea1w+AgPD5eFCxfK559/Llu2bJEf//jHAX/PZC6cM7RcDBo0iNcLjjqHUfUfRERERERERERETYJL9oiIiIiIiIiIqEmxIEVERERERERERE2KBSkiIiIiIiIiImpSLEgREREREREREVGTYkGKiIiIiIiIiIiaFAtSRERE52n16tX48ssvzZ4GXYL279+P//znP2ZPg2yO1ygiIrIiFqSIiCzm7rvvhohARDBkyJBaj33++ec9xwYSHByMY8eOQUQwb968en3emqO8vBzHjx/H4sWLkZKS4vWa5ORk9TXaGDduXJ3n3qJFC7z44os4ePAgSktLcezYMfzzn/9EmzZt6nxtTZ07d8amTZsgIrj77rsDHjdq1Ch88sknOHXqFEpLS3Hw4EH84x//QIcOHer9uSIiIvA///M/WLVqVa3Hub8OCxYsqPW4nj17eo71nXtQUBBmzJiBTz/9FAcOHEBxcTGKioqwb98+zJ8/Hz169PA6/kK/P4HExMTg+eefx6FDh1BcXIwffvgBf/7znxEREeE5JlC+ao709HSvjztq1CisWLECWVlZnvdn3rx5aNu2rd8crr/+eixZsgTZ2dkoKytDZmYm3n33Xb+viaaur9Pp06f9XhMWFobf/OY32LdvH4qLi3HkyBH8/e9/R3x8fL2/bpdffjkuv/zyWrNTc26///3va/14Y8eO9Rw7cODAgMelpaVBRLBnz556fd6LmZ0xY8YgIyMDBQUFKC4uxq5du/CrX/0KISEhtb6uWbNm2LNnT8BzTU5Oxptvvonjx4978rBgwQJcfvnlfse2atUKL7/8Mo4cOYKysjJkZ2dj8eLF6NOnj/pxX3vtNc/7furUKaxcuRLDhw+v9zkPGzYMK1euxOnTp1FUVIQvv/wy4HWqId8DgTT0GqX9DMjOzsZHH32EkSNH+r3umWee8RzbuXPnWj/H4sWLISI4cOCA33NXXHEF/va3v2HHjh04efIkysrKcPLkSaxduxY///nP/TIxf/78euVz27Zt9fgq6a666iq88847yMzM9GRj6dKlfj8LASAqKgq/+93vsH//fpSWluLUqVP45JNPMHjw4Hp/vu7du+O9995DdnY2SkpKsHfvXjz99NMIDQ31O7Zfv35YsmQJcnJyUFJSgm+//RYPP/wwgoL45x0RXTpq/2lPRESmOXfuHKZNm4bVq1erzwcHB2PKlCkoLy+v9Y+3sWPH4rLLLoPL5cIdd9yBhx9+GEVFRQGPf+edd7B06VLP/0dGRqJbt2647777MHbsWEydOhXvvvuu12u2bduGOXPm1Ho+mzdvrvX58PBwrFmzBt26dcOrr76KrVu3okuXLnj00UcxePBgXHvttThz5kytHwMApk2bhpdffrnO42bOnIl//OMf2LNnD5577jnk5OSgd+/euP/++zF+/Hj069cPhw8frvPjpKamIjw8vM4/9oDK93TChAmIjo5Gfn6+eszdd9+tvqeGYWDx4sUYN24cVq1ahRdeeAEnTpxAXFwcBgwYgMmTJ+OOO+7AiBEjsHbtWq/XXoj3J5Do6GisX78enTp1wgsvvIB9+/ZhyJAheOyxx9C7d2/cfPPNAID09HTceuut6sdo164dXnzxRXzzzTeexx566CG89NJL2LJlC5577jkUFhZi4MCB+OlPf4oRI0agd+/enkLRiBEjsHz5cpw4cQIvvPACjhw5gq5du+LBBx/EuHHjkJqaWq/ukPT0dLz22mt+j5eVlXn9f3BwMD766CMMHDgQr776Kr788kv069cPDz74IG644Qb06dMH586dq/PzDRs2DADqnZ277roLv/nNbwIWoANlp6bExESMGzcOLpcLV155JW688UasW7cu4PEXMztPPvkk/vCHP2DTpk147LHHUF5ejsmTJ+OPf/wjevXqhcmTJwd87dNPP40rr7xSfe7KK6/E5s2bUV5ejldffRXff/89+vbti/vvvx/Dhw9Hnz59cOzYMQBAQkICvvzyS8THx+P111/H9u3b0bVrVzz00EMYPnw4UlJS8PXXXwOoLJisX78eYWFheO2117B3714kJyfj5z//OVasWIHbbrsN77//fq3nPG3aNLzxxhvIysrC//7v/yIrKwtTp07Fm2++iTZt2uBPf/qT59iGfA/UpiHXqGPHjuEXv/iF12PNmjVD165dce+99+Ljjz/GL3/5S7zwwgt+r3X/zHr66afVj92yZUuMHj0a5eXlfs/dcMMNWLFiBUpKSjB//nzs3LkTLpcLHTp0wJQpU/Dyyy9j1KhRakHsySefxHfffRfwnOrzc0PTu3dvrF+/HmVlZXj11Vexb98+tG/fHrNmzcLatWsxfvx4fPjhhwAqf36tX78eV199NebPn4+MjAwkJSXhF7/4BVauXIkxY8bgv//9b62f76qrrsKGDRtQXFyMv/zlLzh69CgGDRqE3/72t+jbty8mTJjgOXbYsGH48MMPUVRUhJdeegnff/89xo4di+effx5du3bF/fff36hzJiIyg3BwcHBwWGfcfffdIiKSnp4uBQUFEh0drR43evRoERFZt26dSOVfqOpYsWKFuFwuefXVV0VEZMaMGbV+3ieeeEJ9vmPHjpKXlyc5OTkSEhIiACQ5OVlERP773/+e93n/6le/EhGR+++/3+vxcePGiYjIX//61zo/xsyZM0VE5KWXXvL899133+13nGEYkp2dLXl5eRIfH+/13L333isiIs8//3y95j137lwpLi6W8PDwWo9zv6ciIvfcc496THBwsGRlZXne05pzHzlypIiIpKWlqa8dMmSIiIhs3LjR89iFfH9qO3+XyyWpqalej8+bN092794tV1xxRZ0fY8mSJZKTkyMtW7YUAJKYmCilpaWybds2CQ0N9Tr2hRdeEBGR2bNnex7bunWriIh07drV69ibb75ZREQWL15c6+d3f53mz59fr3OeNWuWiIjcddddXo8/9dRTsn//frnhhhvq9XHee+89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)
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[92\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[94\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
# Test Forecast
future = [
          forecastor(df[:400],
                      model = path,
                      forecast_range_min = 150,
                      num_of_lag = 10).Forecasts.values for path in better_forecasts_paths
         ]
```
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
``` {style="position: relative;"}
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lr_model_2022_6_26-11:24:45.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lasso_model_2022_6_26-11:24:48.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ridge_model_2022_6_26-11:24:51.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lar_model_2022_6_26-11:24:57.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/omp_model_2022_6_26-11:25:1.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/br_model_2022_6_26-11:25:4.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/ransac_model_2022_6_26-11:25:21.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/tr_model_2022_6_26-11:26:54.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
[LOG]Loading Model /content/drive/MyDrive/BIG_DATA_PROJECT/Model_Demo/lightgbm_model_2022_6_26-11:32:58.pkl
Transformation Pipeline and Model Successfully Loaded
[SUCCESS]Model Loaded Successfully
[LOG]Obtaining Data
[SUCCESS]Data Successfully Obtained
[LOG]Forecasting Begining
[SUCCESS]Forecasting Successfully Done
[LOG]Creating Forecast Dataframe
[SUCCESS]Forecasting Dataframe Successfully Created
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Forecasting Future of BTC - EUR Compharison[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Forecasting-Future-of-BTC---EUR-Compharison){.anchor-link} {#Forecasting-Future-of-BTC---EUR-Compharison}
==============================================================================================================================================================================================================================
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[102\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
excellent_future = []
for i in range(150):
    excellent_future.append(np.mean([float(j[i]) for j in future]))
```
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[106\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
``` {style="position: relative;"}
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
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_png .output_subarea}
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABKQAAAJrCAYAAADeTUuLAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzde3hU5b3+//fKORBCOJ8JiICAeEAElbMgiqhoLYiiglW/iuwq1VrtrhXtT+u3tbrVb1XUKmoVthEsqIBSaZCTHKxSBBQVIoicAiTkfJjk+f2RzCQhCTnNkKz13K/rmusiM2tm1mRu3Ls3n+dZDmAQERERERERERE5RcIa+wRERERERERERMQuKqREREREREREROSUUiElIiIiIiIiIiKnlAopERERERERERE5pVRIiYiIiIiIiIjIKaVCSkRERERERERETikVUiIiIiJNSEpKCikpKY19GiIiIiIhpUJKRETEhaZPn44xpsbb9OnT6/X606ZNY9SoUUE+6+C78cYbSU9PxxhT5eOjRo066e/nyy+/rPScyy+/nE8//ZSMjAyys7PZuHEjU6dOrdX5lH+/W2+99aTH3n333YFjExMTA/fPnDmTmTNn1ur9GmrYsGH1zkgwzJs3r1Y5Lv/7sY1b/i6KiIjUVURjn4CIiIjU31tvvcXixYurfXzz5s31et3HH3+cefPm8emnn9b31EIqPj6eF198kalTp5Kbm1vj8UlJSSQlJVW6Pz09vcLPN954I2+88QZbtmzh/vvvJz8/n5tuuokFCxbQoUMHnn322VqdX2FhITNmzODVV1+t9pjp06fj8/mIiKj4/4599NFHtXqPYLj99ttJTEzkjTfeOGXvWZXf/va3fPfdd9U+fvjw4VN4Nk1LU/+7KCIiUl8qpERERFxs27ZtLFq0KKiv2a5duyY/kbJ8+XL69OnDlVdeyf3338/o0aNPevyOHTtq/D3Fxsby7LPPsmfPHkaMGEFOTg4Ab775Jhs3buSJJ55g/vz5pKam1nh+69atY/To0Zx++ul8//33lR4fMGAAgwYNYu3atQwfPrzG1wuV888/v0mUPcnJyWzcuLGxT6PJccPfRRERkfrSkj0REREL+JdGVfU/bnNzcwN7Fs2ZMydQUDzyyCMVlv0ZY0hOTq70/Ouuuw5jDHPmzAncl5ycTGFhIT169GDNmjXk5OQwYMCAwOPDhw9n6dKlHDt2jNzcXL777jv+/Oc/k5CQUKvPs23bNs455xyWLVtW+19CDa688kpat27N3/72t0AZBVBcXMzcuXOJjY1l8uTJtXqt5cuXU1RUxIwZM6p8fPr06aSnp7N69epKj524h5R/eea0adO48sor2bx5M9nZ2Rw7doz//d//pU2bNpWOrWoZ3vLlywMZ8C8t7N+/P6NHj8YYw7x58wLHtmnThmeffZaUlBTy8/NJTU1l8eLFDBkypNLrzpgxg88++4zU1FSys7P57rvvePrpp2v9XdZV3759mT9/PgcOHKCgoICDBw+SlJTEmWeeWeE4f+bPPPNMli1bRlZWFpdffnng8YEDB5KUlMThw4fJz89nz549zJ07l86dO1d6z9NOO423336bgwcPkpeXx/bt27n99ttxHKfCcePGjWP58uUcPXqUvLw8vv/+e1555RU6depU6TVr+r2d7O+iiIiIF2hCSkRERAKSkpIwxvDoo48GlrnVd9kfwPPPP8/69et54YUXOHjwIACTJk1i4cKFfPXVV8yZM4eMjAwuuOAC7rnnHi699FKGDh1KXl7eSV/3jjvuqNf5hIWFERUVVeXr+8uWzz77rNJj/umdoUOH8sILL9T4PocOHeLTTz/lpptu4ve//32FPa7CwsK44YYbeP/99ykoKKj1uV922WVcfPHF/PWvf+Wvf/0rV1xxBddddx1RUVH87Gc/q/XrAGzfvp2f//znLFy4kO3btzNnzhx++OEHABISEvjss89o164dL730Etu2baNLly7MnDmT1atXM2HChEAx+Zvf/IY//elPfPTRRzz44IPk5eUxaNAgZs2axciRIxk8eHCdzqsmAwYMYN26dfh8Pl588UW+/fZbevbsyaxZs/jss88YPnw4//nPfyo8549//CP79+/ntttuY/v27UDJd52cnMxPP/3Ek08+yYEDBzjrrLOYOXMmEydOZPDgwRw6dAgoKaO++OIL0tLSeOKJJ0hLS+Pqq6/m5Zdfpk+fPtx///0ATJgwgQ8++ICvv/6ahx9+mLS0NM466yzuvvtuLrnkEgYMGEB2dnatf2/B/rsoIiLSFBnddNNNN910081dt+nTpxtjjHnggQdqdfy8efOMMcYkJiZWeiw3N9ekpKQEfh41apQxxpg5c+ZUOM4YY5KT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wv2xsbHcc889FBcXs3jxYgCWLVvG/v37ue222ypM4URFRTFr1izS0tIqbdJeG/7leX/5y18oLCxkwYIFDfhEJ1fd9/zzn/+czp07Vzq+qKio0veclJREZGRkpWwkJCSwZcsWli1bBkDnzp3ZunUrf/jDHyq9rn9pY35+fv0/zAlyc3NZunQpAwcOrFQmjRw5kjPOOIMPPvggUMpVZ9euXXz55ZecffbZjB07tsJjQ4YM4eDBgzzwwANAyTLJFStW0L9/fy688MIKx/75z3/mtddeA8qm7cpvXA4lm+n7l+P5/97U5fdW3d9FERERL9CSPREREQu8//77PPTQQzz11FO0b9+e9PR0Lr30Urp27UpKSkqFCaIffviB4uJipk2bxpEjR9i6dSv//Oc/Wbx4MXfffTeLFy9mwYIFtGzZkjvvvJOVK1dy44031uo8CgsLmTlzJklJSaxatYqnn36aAwcOMHjwYO644w527tzJhx9+eNLXaNu2LaNGjQr87J/kufbaawP3bd68mb179/Loo48yceJEXn31VS644AI2b95M+/btue222+jbty+PPPJIYO+iwsJC7rrrLhYtWsSaNWt48cUX8fl83HrrrfTt25fp06fXuBysKnv37iU5OZmxY8eyaNEi0tLS6vwatbV+/XoOHz7M9OnTOXLkCDt27OCss85i2rRpLF++nAkTJlQ4PiUlhfPOO485c+awd+9e5s2bx2OPPcakSZP47//+bzp06MCnn35Khw4duPPOO+nQoQO33XYbAPv372fv3r387ne/IzExkVWrVpGXl0ffvn25++67OXDgQL0KvJO5//77GTlyJEuWLOG5554jJSWF3r17M2vWLFJTUwNFUk3uuusuVq5cyXvvvcfTTz/N999/T79+/Zg1axaHDh3i7bffrvCew4YNY8mSJTz55JMcPHiQK664giuuuIK//OUvgQm7jRs3MmbMGJ599lk2bdrEueeey0033cT06dNZunQp1157LV999RVJSUm1/r1V93dRRETEK4xuuummm2666eau2/Tp040xxjzwwAO1fs4111xjvvjiC5OVlWUOHjxoXnnlFRMfH2++++47k5KSUuHYP/zhDyY9Pd1kZGSYe++91wCmWbNm5umnnzYpKSkmJyfHbN++3dx6661m2LBhxhhj5syZE3h+cnKyMcaYDh06VHkuI0aM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)
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Thank You :D[¶](https://htmtopdf.herokuapp.com/ipynbviewer/temp/5b039b37d4db24269530bd70943ce935/BigDataProject.html?t=1656255816690#Thank-You-:D){.anchor-link} {#Thank-You-:D}
================================================================================================================================================================
:::
:::
:::
