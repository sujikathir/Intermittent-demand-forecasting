# Sales-Forecasting
### Predicting the Sales using Time-series forecasting for month-wise data.

Accurate forecasting of spare parts demand not only *minimizes inventory cost* it also *reduces the risk of stock-out*. Though we have many techniques to forecast demand, majority of them cannot be applied to spare parts demand forecasting. Spare parts demand data usually have many zeros which makes conventional forecasting methods less effective. In this study we have used latest parametric time series method **(FB Prophet)** to forecast spare parts demand of a spare parts company. We have shown that with careful selection of the algorithm and their parameters the FB Prophet library gives accurate forecasts for spare parts demand. Applying the proposed forecasting methods in sales of spare parts, the maintenance and repair companies will reduce inventory costs. The proposed model is trained and forecasted using FB Prophet, launched by Facebook as an API for carrying out the forecasting related things for time series data. The historical univariate data used for the research is a monthly spare parts sales of a Singaporean company. The proposed system is compared with already existing models for time-series analysis such as *ARIMA*, *Moving average*, *Exponential Smoothing* and *Croston*, where the accuracy is minimal compared to FB Prophet. 


### Data science tools used for the project:
1.	**Pandas** is a high performance, easy-to-use and convenient data structure and an analysis tool for python programming language. Pandas provide us a data frame to store the data in a clear way.
2.	**NumPy** is a python library also known as Numeric python which can perform scientific computing. All one must know is that python never provides an array data structure, only with the help of a numpy library it is possible to create and perform manipulations on an array.
3.	**Matplotlib** is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter.


We can also visualize our data using a method called *time-series decomposition* that allows us to decompose our time series into three distinct components: trend, seasonality, and noise. 

1.	**Trend:** The optional and often linear increasing or decreasing behavior of the series over time.
2.	**Seasonality:** The optional repeating patterns or cycles of behavior over time.
3.	**Noise:** The optional variability in the observations that cannot be explained by the model.

![Trend](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/trend%20graph.PNG)

Assumptions can be made about these components both in behavior and in how they are combined, which allows them to be modelled using traditional statistical methods. These components may also be the most effective way to make predictions about future values, but not always.

1.	**Frequency:** Perhaps data is provided at a frequency that is too high to model or is unevenly spaced through time requiring resampling for use in some models.
2.	**Outliers:** Perhaps there are corrupt or extreme outlier values that need to be identified and handled.
3.	**Missing** Perhaps there are gaps or missing data that need to be interpolated or imputed.

Often time series problems are real-time, continually providing new opportunities for prediction. This adds an honesty to time series forecasting that quickly flushes out bad assumptions and errors in modelling.

##Actual Plot
![Actual plot](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/actual%20dataset%20plot.PNG)

### Dealing with Missing values:
![missing](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/filled%20missing%20values%20with%20zeroes.PNG)


*Stationarity* is defined using very strict criterion. However, for practical purposes we can assume the series to be stationary if it has constant statistical properties over time, i.e,
1.	**constant mean**
2.	**constant variance**
3.	**An autocovariance that does not depend on time.**

![Residual mean](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/Seasonal_Decompose.PNG)

## Forecasting using ARIMA Model:

*ARIMA* stands for Auto-Regressive Integrated Moving Averages. The ARIMA forecasting for a stationary time series is nothing but a linear (like a linear regression) equation. The predictors depend on the parameters (p,d,q) of the ARIMA model:

1.	*Number of AR (Auto-Regressive) terms (p):* AR terms are just lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).
2.	*Number of MA (Moving Average) terms (q):* MA terms are lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.
3.	*Number of Differences (d):* These are the number of nonseasonal differences, i.e. in this case we took the first order difference. So either we can pass that variable and put d=0 or pass the original variable and put d=1. Both will generate same results.

An importance concern here is how to determine the value of ‘p’ and ‘q’. Two plots can be used to determine these numbers. They are:

1. **Autocorrelation Function (ACF):** It is a measure of the correlation between the TS with a lagged version of itself. For instance at lag 5, ACF would compare series at time instant ‘t1’…’t2’ with series at instant ‘t1-5’…’t2-5’ (t1-5 and t2 being end points).

![Autocorrelation](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/auto%20correlation%20plot.PNG)

2. **Partial Autocorrelation Function (PACF):** This measures the correlation between the TS with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons. Eg at lag 5, it will check the correlation but remove the effects already explained by lags 1 to 4.


The equation denoted below is for ARIMA model. It is a combination of AR model and MA model.

![Eqn Arima](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/done%20Eqn%20for%20ARIMA.png)

**Stationary Series**
One of the requirements for ARIMA is that the time series should be stationary. A stationary series is one where the properties do not change over time. There are several methods to check the stationarity of a series. The one you’ll use in this project is the Augmented Dickey-Fuller test.

**Augmented Dickey-Fuller Test**
The Augmented Dickey-Fuller test is a type of statistical unit root test. The test uses an autoregressive model and optimizes an information criterion across multiple different lag values.

![Augmented test](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/done%20Augmented%20dicker-fuller%20test.PNG)

The null hypothesis of the test is that the time series is not stationary, while the alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.

The first step is to import the adfuller module from the statsmodels package. This is done in the first line of code below. The second line performs and prints the p-value of the test.

The output above shows that the p-value is greater than the significance level of 0.05, so we fail to reject the null hypothesis. The series is not stationary and requires differencing.

![differenciate](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/done%20Diff%20fxn.PNG
)

The series can be differenced using the diff() function. The first line of code above performs the first order differencing, while the second line performs the Augmented Dickey-Fuller Test.


It is recommended to use the *auto_arima* function in Python, which automatically discovers the optimal order for an ARIMA model. In simple terms, the function will automatically determine the parameters p, d, and q of the ARIMA model.

It is necessary now to build the ARIMA estimator. The first step is to import the *pmdarima library* that contains the *auto_arima* function. 

The time-series to fit the ARIMA model contains parameters as:
1.	*start_p:* the starting value of p, the order of the auto-regressive (AR) model. This must be a positive integer.
2.	*start_q:* the starting value of q, the order of the moving-average (MA) model. This must be a positive integer.
3.	*d:* the order of first-differencing. The default setting is none, and then the value is selected automatically based on the results of the test, in this case the Augmented Dickey-Fuller test.
4.	*test:* type of unit root test to use in order to detect stationarity if stationary is False and d is none.

The second step is to define a function that takes in the time series array and returns the auto-arima model. 

![Auto_arima](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/done%20best%20model%20for%20arima.PNG)

The output above shows that the final model fitted was an ARIMA(0,1,0) estimator, where the values of the parameters p, d, and q were zero, one, and zero, respectively. The auto_arima functions tests the time series with different combinations of p, d, and q using AIC as the criterion. 

AIC stands for Akaike Information Criterion, which estimates the relative amount of information lost by a given model. In simple terms, a lower AIC value is preferred. In the above output, the lowest AIC value of 89.575 was obtained with a Total Fit time of 10.597 seconds for the ARIMA(0, 1, 0) model, and that is used as the final estimator.

The model is trained and will now use it make predictions on the test data and perform model evaluation. 

The predicted graph above shows that ARIMA doesn’t perform well for the intermittent univariate dataset with more zero/null values. 

## Moving average model

This model still forms the basis of many time series decomposition methods, so it is important to understand how it works. The first step in a classical decomposition is to use a moving average method to estimate the trend-cycle.

The estimate of the trend-cycle at time t is obtained by averaging values of the time series within k periods of t. Observations that are nearby in time are also likely to be close in value. Therefore, the average eliminates some of the randomness in the data, leaving a smooth trend-cycle component. We call this an m-MA, meaning a moving average of order m.

![Moving avg](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/done%20Moving%20average%20model.PNG)

The output graph consists of the Rolling mean trend with the upper bound and lower bound.

The order of the moving average determines the smoothness of the trend-cycle estimate. In general, a larger order means a smoother curve.

## Exponential Smoothing

Forecasts produced using exponential smoothing methods are weighted averages of past observations, with the weights decaying exponentially as the observations get older. The simplest of the exponentially smoothing methods is naturally called simple exponential smoothing (SES). This method is suitable for forecasting data with no clear trend or seasonal pattern.


We often want something between these two extremes. For example, it may be sensible to attach larger weights to more recent observations than to observations from the distant past. This is exactly the concept behind simple exponential smoothing. Forecasts are calculated using weighted averages, where the weights decrease exponentially as observations come from further in the past — the smallest weights are associated with the oldest observations:

![Exp](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/done%20exponential%20eqn.PNG)

The one-step-ahead forecast for time T+1 is a weighted average of all of the observations in the series y1,…,yT. The rate at which the weights decrease is controlled by the parameter α.

For any α between 0 and 1, the weights attached to the observations decrease exponentially as we go back in time, hence the name “exponential smoothing.” If α is small (i.e., close to 0), more weight is given to observations from the more distant past. If α is large (i.e., close to 1), more weight is given to the more recent observations. For the extreme case where α=1 forecasts are equal to the naïve forecasts.

**Predicted graph when α = 0.5 (Most optimised value)**

![Exp smooth graph](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/exponential%20smoothing%20graph.PNG)

It can be seen that exponential smoothing doesn’t fit well for the given dataset.

## Croston’s Method 
Croston method is the most frequently used technique for sporadic demand forecasting. In Croston’s algorithm, the historical demand is separated into two series: one representing the non-zero demand and the other representing inter-arrival time. The inter-arrival time is identified as the period between two consecutive non-zero demands. 

Croston method forecasts the non-zero demand size and the inter-arrival time between successive demands using exponential smoothing individually. Both forecasts are updated only after demand occurrences. 

We consider the following notation: Y(t) is the estimate of the mean size of a nonzero demand at time t, P(t) is the estimate of the mean interval between nonzero demands at time t, X(t) is the actual demand at time t, Q is the time interval since the last nonzero demand and α is the smoothing constant.

Croston forecasting method updates values of Y(t) and P(t)according to the procedure shown in figure:

![crost](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/croston%20flowchart.PNG)

Output of croston model representing Actual demand vs Predicted/Forecasted Demand.

![crost output](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/done%20Croston%20actual%20vs%20predicted%20table.PNG)

## Forecasting using FB PROPHET

The ARIMA approach works decently well with stationary data and when forecasting short time frames, but Facebook’s engineers have built a tool for those cases which ARIMA can’t handle. Prophet is built with its backend in STAN, a probabilistic coding language. This allows Prophet to have many of the advantages offered by Bayesian statistics, including seasonality, the inclusion of domain knowledge, and confidence intervals to add a data-driven estimate of risk.

Prophet follows the sklearn model API. It allows to create an instance of the Prophet class and then call its fit and predict methods. Prophet requires time series data to have a minimum of two columns: ds which is the time stamp and y which is the values. After loading our data, we need to format it as such:

![fb](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/done%20ds%2C%20y.PNG)

The model is fit by instantiating a new Prophet object. Any settings to the forecasting procedure are passed into the constructor. Then call its fit method and pass in the historical dataframe. Fitting should take 1-5 seconds.

**Using Date-time parser:**

To convert the data into required format for the prophet, we make the following modification:

1.	**parse_dates:** This specifies the column which contains the date-time information. As we say above, the column name is ‘Month’.
2.	**index_col:** A key idea behind using Pandas for TS data is that the index has to be the variable depicting date-time information. So this argument tells pandas to use the ‘Month’ column as index.
3.	**date_parser:** This specifies a function which converts an input string into datetime variable. Be default Pandas reads data in format ‘YYYY-MM-DD HH:MM:SS’. If the data is not in this format, the format has to be manually defined. Something similar to the dataparse function defined here can be used for this purpose.

With just a few lines, Prophet can make a forecast model every bit as sophisticated as the ARIMA model built previously. 

Here, this project uses Prophet to make a 3-year forecast (frequency is monthly, periods are 12 months/year times 3 years):

1.	Importing Prophet library and necessary tools
2.	Predictions are then made on a dataframe with a column ds containing the dates for which a prediction is to be made.

 You can get a suitable dataframe that extends into the future a specified number of days using the helper method Prophet.make_future_dataframe. By default it will also include the dates from the history, so we will see the model fit as well.

The important idea in Prophet is that by doing a better job of fitting the trend component very flexibly, we more accurately model seasonality and the result is a more accurate forecast. We prefer to use a very flexible regression model (somewhat like curve-fitting) instead of a traditional time series model for this task because it gives us more modelling flexibility, makes it easier to fit the model, and handles missing data or outliers more gracefully.

By default, Prophet will provide uncertainty intervals for the trend component by simulating future trend changes to your time series. To check model uncertainty about future seasonality or holiday effects, it is possible by running a few hundred HMC iterations (which takes a few minutes) and the forecasts will include seasonal uncertainty estimates.

Prophet model is fit using Stan, and have implemented the core of the Prophet procedure in Stan’s probabilistic programming language. Stan performs the MAP optimization for parameters extremely quickly (<1 second), gives us the option to estimate parameter uncertainty using the Hamiltonian Monte Carlo algorithm, and allows to re-use the fitting procedure across multiple interface languages. Currently it provides implementations of Prophet in both Python and R. 

Output for FB Prophet:
•	The predict method will assign each row in future a predicted value which it names yhat. If the historical dates are passed, it will provide an in-sample fit. The forecast object here is a new dataframe that includes a column yhat with the forecast, as well as columns for components and uncertainty intervals.

![prophet](https://github.com/sujikathir/Intermittent-demand-forecasting/blob/main/Images/graph%20output%20fbprophet.PNG)

