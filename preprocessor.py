import pandas as pd
import pyarrow.parquet as pq

class preprocessing:
    def df_pp(path,freq):
        try:
            print(f'[\033[96mLOG\033[0m]Data {path} Reading')
            df = pq.read_table(source=path).to_pandas()
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
