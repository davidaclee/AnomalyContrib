import pandas as pd
import numpy as np
from read_data import *
from anomaly_detect import *
from factor_form import *
# 
if __name__ == '__main__':
    """
    日期参数
    input_date: 指标观察日期；
    base_interval: 以 n 天前作为基线日期，该值为 1，即环比昨天，该值为 7 即周同比上周；
    hist_interval: 往前追溯多少天
    """
    input_date = '20210421'
    base_interval = 7
    hist_interval = 7


    """
    mstag: MySQL 连接标识符
    tbl_name: 连接表名
    dt_col: 日期字段名称
    """

    mstag = 'test'
    tbl_name = 'demo_onecloud'
    dt_col = 'dayno'
    stat_metric = ['req_nums', 'req_fill_nums', 'expose_nums', 'acc_nums', 'acc_cost']
    comp_metric = ['req_nums', 'req_fill_rate', 'expose_rate', 'ctr', 'ppc']

    obs_metric = 'acc_cost'
    comp_metric.append(obs_metric)
    dim_col = ['ad_show_pos', 'ad_owner_id']

    stat_date, base_date, hist_date = date_gen(input_date, base_interval, hist_interval, False)

    path = '/Users/davidaclee/Desktop/data_coding/data_project/AnomalyContrib/data/test.csv' 
    sql1= 'select * from {} where 1 = 1 and dt between "{}" and "{}"'.format(tbl_name, hist_date, stat_date)
    # df_ = load_from_mysql(mstag, sql1)
    df_ = load_from_csv(path)
    # logging.info(df_)
    df_[dt_col] = df_[dt_col].astype(str)
    df_[dim_col] = df_[dim_col].astype(str)
    
    df = df_.query('{} in ["{}", "{}"]'.format(dt_col, stat_date, base_date))
    df_hist = df_.groupby(by = dt_col)[obs_metric].sum().reset_index()
    

    logging.info(df)
    logging.info(df.info())
    logging.info(df_hist)
    
    detect_info_dod = dod_threshold_detect(df_hist, dt_col, stat_date, base_date, obs_metric)
    detect_info_ksigma = k_sigma(3, df_hist, dt_col, stat_date, base_date, obs_metric)
    detect_info_boxplot = boxplot_detect(df_hist, dt_col, stat_date, base_date, obs_metric)
    detect_info_zscore = zscore_detect(df_hist, dt_col, stat_date, base_date, obs_metric)

    # # logging.info([detect_info_dod['detect_result'], detect_info_ksigma['detect_result'], detect_info_boxplot['detect_result'], detect_info_zscore['detect_result']])
    logging.info([detect_info_dod, detect_info_ksigma, detect_info_boxplot, detect_info_zscore])

    # df_factor_form_t = get_factor_form(df, dt_col, stat_date, base_date, obs_metric, stat_metric, comp_metric)
    # logging.info('df_factor_form_t:\n{}'.format(df_factor_form_t))
    # # df_factor_form_t = df_factor_form.T
    # # df_factor_form_t['impact_' + obs_metric] = 0
    # # df_factor_form_t['diff_pct'] = df_factor_form_t[stat_date]/df_factor_form_t[base_date]
    # # df_factor_form_t['diff_value'] = df_factor_form_t[stat_date]-df_factor_form_t[base_date]

    # # logging.info('df_factor_form_t:\n', df_factor_form_t)