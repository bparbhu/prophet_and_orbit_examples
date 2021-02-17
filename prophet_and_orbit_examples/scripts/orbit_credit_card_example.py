import pandas as pd
import numpy as np
import time
import cmdstanpy as cstanpy
import pystan
import d6tflow
import d6tcollect
import arviz
import matplotlib.pyplot as plt
from dask.distributed import Client
import datetime as dt
from datetime import timedelta
from orbit.models.dlt import DLTFull, DLTMAP
from orbit.diagnostics.plot import plot_predicted_data, plot_posterior_params
import itertools
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, RangeTool, DataRange1d, Select

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth',None)

if __name__ == '__main__':

    # This script should be a showcase of the Bayesian workflow using orbit

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
    msk = (fred_data.loc[:, 'type'] == 'us_retail_advanced_sa')
    us_retail_adv_sa = fred_data.loc[msk, :]

    # let's look at what the percentage change is month over month
    us_retail_adv_sa.loc[:, 'pct_chg_in_sales_from_prev_mnth'] = \
        us_retail_adv_sa.groupby(['type'])['value'].apply(
            lambda x: x.pct_change())

    us_retail_adv_sa.loc[:, 'pct_chg_in_sales_from_prev_mnth'] = \
        us_retail_adv_sa.loc[:, 'pct_chg_in_sales_from_prev_mnth'] * 100

    short_us_retail = us_retail_adv_sa[['date', 'pct_chg_in_sales_from_prev_mnth']]

    # So let's aggregate the credit dataset to monthly sales
    credit_agg_data = credit_card_data.groupby(['year', 'month'], as_index=False).agg({
        'panel_sales': 'sum',
        'transaction_count':'sum',
    })

    # so let's compute the percent in change of panel sales from month to month
    credit_agg_data['pct_chg_in_sales_from_prev_mnth'] = \
        credit_agg_data['panel_sales'].pct_change(fill_method='ffill')

    credit_agg_data.loc[:, 'pct_chg_in_sales_from_prev_mnth'] = \
        credit_agg_data.loc[:, 'pct_chg_in_sales_from_prev_mnth'] * 100

    credit_agg_data.loc[:,'day'] = 1 # this is just to convert the column to datetime and look at the data monthly

    credit_agg_data.loc[:,'date'] = \
        pd.to_datetime((credit_agg_data.year*10000+credit_agg_data.month*100+credit_agg_data.day).apply(str),
                       format='%Y%m%d')

    clms = ['date','panel_sales','transaction_count','pct_chg_in_sales_from_prev_mnth']
    credit_agg_short = credit_agg_data[clms]

    msk = (credit_agg_short.loc[:,'pct_chg_in_sales_from_prev_mnth'].isna())
    credit_agg_short.loc[msk,'pct_chg_in_sales_from_prev_mnth'] = 0

    test_size = 14
    train_df = credit_agg_short[:-test_size]
    test_df = credit_agg_short[-test_size:]

    regressors = ['panel_sales','transaction_count']

    dlt = DLTFull(
        response_col='pct_chg_in_sales_from_prev_mnth',
        regressor_col=regressors,
        date_col='date',
        seasonality=1,
        seed=2020,
        level_sm_input=0.3,  # recommend for higher frequency data
        regressor_sigma_prior=[0.5] * len(regressors),
        regression_penalty='lasso',
        period=365,
        prediction_percentiles=[5,95]
    )

    dlt.fit(df=train_df)

    pystan.check_hmc_diagnostics(dlt)

    density_plot = plot_posterior_params(dlt, kind='density',
                            incl_trend_params=True, incl_smooth_params=True)

    trace_plot = plot_posterior_params(dlt, kind='trace',
                              incl_trend_params=True, incl_smooth_params=True)

    pair_plot = plot_posterior_params(dlt, kind='pair', pair_type='reg',
                          incl_trend_params=False, incl_smooth_params=False)

    num_periods = 12 * 1
    freq = 1
    date_col = dlt.date_col
    last_dt = (dlt.date_col.dt.to_pydatetime())[-1]
    dts = [last_dt + timedelta(days=x * freq) for x in range(1, num_periods + 1)]
    future_df = pd.DataFrame(dts, columns=[date_col])

    predicted_df_dlt = dlt.predict(df=future_df, decompose=True)


    plot_predicted_data(training_actual_df=credit_agg_short[-90:], predicted_df=predicted_df_dlt[-90:],
                        test_actual_df=test_df, date_col=dlt.date_col,
                        actual_col='pct_chg_in_sales_from_prev_mnth', pred_col='predicted_pct_chg_in_sales')

