from ml_model_train import statistical_models
from preprocessor import preprocessing
from feature_extractor import feature_extractor
import warnings
warnings.filterwarnings("ignore")


#model_setup(df, save_path = '/content/drive/MyDrive/BIG_DATA_PROJECT/',target = 'y')
def main():

    df = preprocessing.df_pp('/Users/altemur/Desktop/ITU/Big Data/Big_Data_ML/Stock_Forecasting_Library_by_Altemur-main/BTC-EUR.parquet', freq = 'T')
    df = feature_extractor.create_features(df) 
    df = feature_extractor.series_lagger(df[['y']], full_data = df, n_in = 10, n_out=1, dropnan=True)

    for model in statistical_models.model_pool().keys():
        model_to_train = statistical_models(model)
        model_to_train.model_creator(df, save_path = '/content/drive/MyDrive/BIG_DATA_PROJECT/Models_Deneme/', target = 'y')

if __name__ == '__main__':
    main()