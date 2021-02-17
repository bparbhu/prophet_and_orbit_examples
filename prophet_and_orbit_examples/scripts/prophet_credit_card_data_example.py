import pandas as pd
import numpy as np
import time
import cmdstanpy as cstanpy
import d6tflow
import d6tcollect
import arviz
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
import logging
logger = logging.getLogger("fbprophet")
logger.addHandler(logging.NullHandler())
logger_path = "fbprophet.log"
fh = logging.FileHandler(logger_path, encoding="utf-8")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
import fbprophet as prophet
from fbprophet.plot import plot_plotly, plot_components_plotly, \
    add_changepoints_to_plot, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics
import itertools
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, RangeTool, DataRange1d, Select

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth',None)

def make_time_series_plot(data,timeframe, output_file_name):
    """
    This function should make a ranged plot for looking at month over month or year over year
    percentage changes in sales
    :param timeframe: str: "yearly" or "monthly"
    :param data: dataframe
    :param output_file_name: str:
    :return: month over month or year over year bokeh range plot
    """
    output_file(filename=output_file_name)
    dates = np.array(data['date'], dtype=np.datetime64)

    if timeframe == 'yearly':
        the_plot = figure(title="Year over Year percentage change in retail advanced sales",
                            plot_height=300, plot_width=800, tools="xpan", toolbar_location=None,
                            x_axis_type="datetime", x_axis_location="above", x_range=(dates[150], dates[250]),
                            y_range=None,
                            background_fill_color="#efefef")
    elif timeframe == 'monthly':
        the_plot = figure(title="Month over month percentage change in retail advanced sales",
                        plot_height=300, plot_width=800, tools="xpan", toolbar_location=None,
                        x_axis_type="datetime", x_axis_location="above", x_range=(dates[12], dates[24]),
                        y_range=None,
                        background_fill_color="#efefef")
    else:
        raise AssertionError("Pick a timeframe")

    source = ColumnDataSource(data)

    the_plot.line('date', 'pct_chg_in_sales_from_prev_mnth', source=source)
    the_plot.yaxis.axis_label = 'Percentage Change in Sales'

    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    plot_height=130,
                    plot_width=800,
                    y_range=the_plot.y_range,
                    x_axis_type="datetime",
                    y_axis_type='linear',
                    tools="",
                    toolbar_location=None,
                    background_fill_color="#efefef")

    range_tool = RangeTool(x_range=the_plot.x_range, y_range=None)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line('date', 'pct_chg_in_sales_from_prev_mnth', source=source)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    # let's display our timeseries
    show(column(the_plot, select))

def make_state_wide_time_series_plot(data,timeframe, output_file_name):
    """
    This function should plot the credit card time series by
    state at the month over month or year over year level
    :param data:pandas dataframe
    :param timeframe: str: 'yearly' or 'monthly'
    :param output_file_name: 'str': location where the html file should be sennt
    :return: html plot file
    """

    source = ColumnDataSource(data)
    output_file(output_file_name)

    if timeframe == 'yearly':
        the_plot = figure(title="Year over Year percentage change in credit card sales",
                            plot_height=300, plot_width=800, tools="", toolbar_location=None,
                            x_axis_type="datetime",
                            y_range=None,
                            background_fill_color="#efefef")
    elif timeframe == 'monthly':
        the_plot = figure(title="Month over month percentage change in credit card sales",
                        plot_height=300, plot_width=800, tools="", toolbar_location=None,
                        x_axis_type="datetime",
                        y_range=None,
                        background_fill_color="#efefef")
    else:
        raise AssertionError("Pick a timeframe")

    the_plot.line('date', 'pct_chg_in_sales_from_prev_mnth', source=source)
    the_plot.yaxis.axis_label = 'Percentage Change in Sales'
    the_plot.axis.axis_label_text_font_style = "bold"
    the_plot.x_range = DataRange1d(range_padding=0.0)
    the_plot.grid.grid_line_alpha = 0.3

    state = 'Alaska'
    all_states = dict(zip(credit_agg_data.state,credit_agg_data.pct_chg_in_sales_from_prev_mnth))
    state_select = Select(value=state, title='State', options=sorted(all_states.keys()))
    distribution_select = Select(value=distribution, title='Distribution', options=['Discrete', 'Smoothed'])

def run_cross_val_tuning(data):
    """
    This function should run prophet for the time series we want to fit and the parameters we want to tune
    :param data: dataframe containing ds and y values
    :return: tuning results
    """
    perf_metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'coverage']

    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 10]
    }

    # # Generate all combinations of parameters
    parameters = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    rmses = []  # Store the metrics for each specific parameter here
    mse = []
    mae = []
    mape = []
    mdape = []
    coverage = []

    for params in parameters:
        model = prophet.Prophet(**params).fit(data)
        data.cv = cross_validation(model, period='365 days',
                                   horizon='720 days', parallel='dask')
        data.p = performance_metrics(data.cv, rolling_window=1, metrics=perf_metrics)
        rmses.append(data.p['rmse'].values[0])
        mae.append(data.p['mae'].values[0])
        mdape.append(data.p['mdape'].values[0])
        mse.append(data.p['mse'].values[0])
        mape.append(data.p['mape'].values[0])
        coverage.append(data.p['coverage'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(parameters)
    tuning_results['rmse'] = rmses
    tuning_results['mse'] = mse
    tuning_results['mae'] = mae
    tuning_results['mape'] = mape
    tuning_results['mdape'] = mdape
    tuning_results['coverage'] = coverage

    print('These are your tuning results')
    print(tuning_results)

    best_rmses_params = parameters[np.argmin(rmses)]
    best_mse_params = parameters[np.argmin(mse)]
    best_mae_params = parameters[np.argmin(mae)]
    best_mape_params = parameters[np.argmin(mape)]
    best_mdape_params = parameters[np.argmin(mdape)]
    best_coverage_params = parameters[np.argmin(coverage)]

    print('The Best RMSES paramters are')
    print(best_rmses_params)

    print('The Best MSE paramters are')
    print(best_mse_params)

    print('The Best MAE paramters are')
    print(best_mae_params)

    print('The Best MAPE paramters are')
    print(best_mape_params)

    print('The Best MDAPE paramters are')
    print(best_mdape_params)

    print('The Best parameters for Coverage are')
    print(best_coverage_params)

    return tuning_results

if __name__ == "__main__":

    # read in our data
    credit_card_data = pd.read_parquet("C:/Users/brian/Documents/ubs_interview/credit_card_df.parquet")
    fred_data = pd.read_parquet("C:/Users/brian/Documents/ubs_interview/fred_df.parquet")

    # let's create some features we should already see in our data
    credit_card_data['month'] = pd.DatetimeIndex(credit_card_data['optimized_date']).month
    credit_card_data['year'] = pd.DatetimeIndex(credit_card_data['optimized_date']).year
    credit_card_data['month_name'] = pd.to_datetime(credit_card_data['month'],
                                                    format='%m').dt.month_name().str.slice(stop=3)

    fred_data['month'] = pd.DatetimeIndex(fred_data['date']).month
    fred_data['year'] = pd.DatetimeIndex(fred_data['date']).year
    fred_data['month_name'] = pd.to_datetime(fred_data['month'],
                                                    format='%m').dt.month_name().str.slice(stop=3)
    # let's look at us retail advanced sa only
    msk = (fred_data.loc[:,'type']=='us_retail_advanced_sa')
    us_retail_adv_sa = fred_data.loc[msk,:]

    # let's look at what the percentage change is month over month
    us_retail_adv_sa.loc[:,'pct_chg_in_sales_from_prev_mnth'] =     \
        us_retail_adv_sa.groupby(['type'])['value'].apply(
            lambda x: x.pct_change())

    us_retail_adv_sa.loc[:,'pct_chg_in_sales_from_prev_mnth'] = \
        us_retail_adv_sa.loc[:,'pct_chg_in_sales_from_prev_mnth'] * 100


    # Let's take a look at what the year over year % change in sales look like
    make_time_series_plot(data=us_retail_adv_sa,
                          timeframe='yearly',
                          output_file_name='Year_over_year_retail_adv_sales.html')


    # so we see that 2008 had extreme fluctuations as compared to other years
    # so let's take a look at 2008 and see what might have occured due to the great recession
    msk = (us_retail_adv_sa.loc[:,'year'].isin([2008,2008.0, '2008',2007,'2007',2007.0]))
    great_rec_year = us_retail_adv_sa.loc[msk,:]


    # Though we really want to look at what a single year typically looks like where there isn't a recession
    # so let's look at two years and compare
    # Also let's see what the month over month plot looks like
    make_time_series_plot(data=us_retail_adv_sa,
                          timeframe='monthly',
                          output_file_name='Month_over_month_retail_adv_sales.html')


    # So let's aggregate the credit dataset to monthly sales
    credit_agg_data = credit_card_data.groupby(['state','year','month'], as_index=False).agg({
        'panel_sales':'sum',
    })
    # so let's compute the percent in change of panel sales from month to month
    credit_agg_data['pct_chg_in_sales_from_prev_mnth'] = \
        credit_agg_data.groupby(['state'])['panel_sales'].apply(
            lambda x: x.pct_change(fill_method='ffill'))

    credit_agg_data.loc[:,'pct_chg_in_sales_from_prev_mnth'] = \
        credit_agg_data.loc[:,'pct_chg_in_sales_from_prev_mnth'] * 100

    # let's look at what the month to month pct in change in sales from the prev month looks like

    # let's create a simple prophet model for forecasting month to month pct change from the prev month
    # let's reduce what we need to look at in our fred data

    short_us_retail = us_retail_adv_sa[['date','pct_chg_in_sales_from_prev_mnth']]

    short_us_retail.rename(columns={'date':'ds','pct_chg_in_sales_from_prev_mnth':'y'}, inplace=True)


    simple_model = prophet.Prophet(interval_width=0.95,
                                   changepoint_prior_scale=0.5,
                                   seasonality_mode='multiplicative')

    simple_model.fit(short_us_retail)

    future_dates = simple_model.make_future_dataframe(periods=12, freq='MS')


    # let's check the forecasted dates we're predicting for
    future_dates.tail()

    # let's build our forecasts
    forecast = simple_model.predict(future_dates)

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


    # let's do some visualiztions with our simple timeseries model
    simple_model_plot = simple_model.plot(forecast,uncertainty=True)
    simple_model_plot_w_chgpts = add_changepoints_to_plot(simple_model_plot.gca(),simple_model,forecast)
    plt.show()

    # let's look at the individual components for each of the models we built
    plotly_comp_model_plt = plot_components_plotly(simple_model, forecast)

    # Now let's do cross validation for the models we built and specify the performance metrics and plot them

    perf_metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'coverage']

    client = Client(processes=False)

    simple_model.cv = cross_validation(simple_model,period='365 days',
                                       horizon = '720 days', parallel='dask')

    simple_model.pm = performance_metrics(simple_model.cv,metrics=perf_metrics)


    # let's plot cross validation metrics for the simple model
    mse_plot_simple_model = plot_cross_validation_metric(simple_model.cv, metric = 'mse')
    rmse_plot_simple_model = plot_cross_validation_metric(simple_model.cv, metric = 'rmse')
    mae_plot_simple_model = plot_cross_validation_metric(simple_model.cv, metric = 'mae')
    mape_plot_simple_model = plot_cross_validation_metric(simple_model.cv, metric = 'mape')
    mdape_plot_simple_model = plot_cross_validation_metric(simple_model.cv, metric = 'mdape')
    coverage_plot_simple_model = plot_cross_validation_metric(simple_model.cv, metric = 'coverage')

    # let's tune our model to check what might be the best parameters we can use
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    tuning_results = run_cross_val_tuning(data=short_us_retail)

    print("--- %s seconds ---" % (time.time() - start_time))

