# This Project is Forecastor Project Which PreProcess all Historical Binance Dataset and Create Forecast Models with 25 Algorithm
## BTC-EUR Forecast Table Forecast vs Real
![image](https://user-images.githubusercontent.com/67932543/178141561-eb893385-9b13-4073-bd15-2347a1cce0a7.png)

# Example Inference
```bash
python inference_one.py --df <Path to Parquet File> \
                        --model <Path to Trained Model> \
                        --forecast_range_min <Forecast Range in Mintes> \
                        --num_of_lag <Number of Lags which trained in Model>

python inference_one.py --df BTC_EUR.parquet --model rf_model.pkl --forecast_range_min 30 --num_of_lag 10
```

# Models List:
|Models Shortcut|Model Name|
|-|-|
|lr|Linear Regression|
|lasso|Lasso Regression|
|ridge|Ridge Regression|
|en |Elastic Net|
|lar |Least Angle Regression|
|llar |Lasso Least Angle Regression|
|omp |Orthogonal Matching Pursuit|
|br |Bayesian Ridge|
|ard |Automatic Relevance Determination|
|par |Passive Aggressive Regressor|
|ransac |Random Sample Consensus|
|tr |TheilSen Regressor|
|huber |Huber Regressor|
|kr |Kernel Ridge|
|svm |Support Vector Regression|
|knn |K Neighbours Regressor|
|dt |Decisiopn Tree Regressor|
|rf |Random Forest Regressor|
|et |Extra Trees Regressor|
|ada |AdaBoost Regressor|
|gbr |Gradient Boosting Regressor|
|mlp |MLP Regressor|
|xgboost |Extreme Gradient Boosting|
|lightgbm |Light Gradient Boosting Machine|
|catboost |CatBoost Regressor|

<div class="sc-iznttV DmKWa"><div class="sc-kNjMHG bOLMah"><h2 class="sc-bjUoiL sc-idiyUo sc-hWlEnr fykzbL cAPRbO">About Dataset</h2></div><div class="sc-lkCHeq efwMvM"><div class="sc-itEyKb gxWnca"><div style="min-height: 80px;"><div class="markdown-converter__text--rendered"><h3>Introduction</h3>
<p>This is a collection of 1 minute candlesticks of the top 1000 cryptocurrency pairs on <a rel="noreferrer nofollow" href="https://binance.com">Binance.com</a>. In total there are 1971, but Kaggle limits file count to 1000. Both retrieval and uploading the data is fully automated—see <a rel="noreferrer nofollow" href="https://github.com/gosuto-ai/candlestick_retriever">this GitHub repo</a>.</p>
<h3>Content</h3>
<p>For every trading pair, the following fields from <a rel="noreferrer nofollow" href="https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#klinecandlestick-data">Binance's official API endpoint for historical candlestick data</a> are saved into a Parquet file:</p>
<pre><code> #   Column                        Dtype         
---  ------                        -----         
 0   open_time                     datetime64[ns]
 1   open                          float32       
 2   high                          float32       
 3   low                           float32       
 4   close                         float32       
 5   volume                        float32       
 6   quote_asset_volume            float32       
 7   number_of_trades              uint16        
 8   taker_buy_base_asset_volume   float32       
 9   taker_buy_quote_asset_volume  float32       
dtypes: datetime64[ns](1), float32(8), uint16(1)
</code></pre>
<p>The dataframe is indexed by <code>open_time</code> and sorted from oldest to newest. The first row starts at the first timestamp available on the exchange, which is July 2017 for the longest running pairs.</p>
<p>Here are two simple plots based on a single file; one of the opening price with an added indicator (MA50) and one of the volume and number of trades:<br>
<img alt="" src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2234678%2Fcd04ed586b08c1576a7b67d163ad9889%2Fdownload-1.png?generation=1582053899082078&amp;alt=media"></p>
<h3>Inspiration</h3>
<p>One obvious use-case for this data could be technical analysis by adding indicators such as moving averages, MACD, RSI, etc. Other approaches could include backtesting trading algorithms or computing arbitrage potential with other exchanges.</p>
<h3>License</h3>
<p>This data is being collected automatically from crypto exchange Binance.</p></div></div></div><div class="mdc-drawer-scrim"></div></div></div></div></div>

🥏 Kaggle Dataset Link: https://www.kaggle.com/datasets/jorijnsmit/binance-full-history
## You can check tutorial from BigDataProject.ipynb.
