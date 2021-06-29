import pandas as pd
import numpy as np
from read_data import *

# 同环比阈值法
# k-sigma
# 箱型图
# z-score 转换

# alarm_condition = {
#     'threshold': threshold_detect(a, b)
# }

# if alarm_condition['threshold']:
#     print('数据异常')

def output_detect_msg(detect_type, detect_comment, detect_alg, diff_value, diff_pct, detect_msg):

    detect_info = {
        'detect_type': detect_type,
        'detect_comment': detect_comment,
        'detect_result': detect_alg,
        'diff_value': diff_value,
        'diff_pct': diff_pct,
        'detect_msg': detect_msg
    }

    logging.info(detect_info['detect_msg'])
    return detect_info


def dod_threshold_detect(df, dt_col, stat_date, base_date, obs_metric):

    detect_type = 'dod_thredshold'
    detect_comment = '与基线日期对比，波动百分比是否超过阈值 10%'

    stat_metric = df[df[dt_col].astype(str) == stat_date][obs_metric].values[0]
    base_metric = df[df[dt_col].astype(str) == base_date][obs_metric].values[0]

    diff_value = float(stat_metric) - float(base_metric)
    diff_pct = diff_value/float(base_metric)
    # diff_type = lambda diff_value: '上涨' if diff_value > 0 else '下降'
    if diff_value > 0:
        diff_type = '上涨'
    else:
        diff_type = '下降'

    detect_alg = abs(diff_pct) > 0.1

    if detect_alg:
        detect_msg = '{} 检测数据异常\n观察组 {} 的 {} 值为 {:.2f}，对比基期 {} {} {:.2%}'.format(detect_type,stat_date, obs_metric, stat_metric, base_date, diff_type, diff_pct)
    else:
        detect_msg = '{} 检测数据正常, have a nice day :-)'.format(detect_type)

    detect_info = output_detect_msg(detect_type, detect_comment, detect_alg, diff_value, diff_pct, detect_msg)
    return detect_info
    

def k_sigma(k, df, dt_col, stat_date, base_date, obs_metric):

    detect_type = 'k_sigma'
    detect_comment = '引入统计学对符合正态分布的数据集判断异常值方法来判断指标是否异常'

    stat_metric = df[df[dt_col].astype(str) == stat_date][obs_metric].values[0]
    df[obs_metric] = df[obs_metric].astype('float')
    k = k
    ymean = np.mean(df[obs_metric])
    ystd = np.std(df[obs_metric])

    threshold1 = ymean - k * ystd
    threshold2 = ymean + k * ystd

    if stat_metric < threshold1:
        diff_value = stat_metric - threshold1
        diff_pct = diff_value/threshold1
    elif stat_metric > threshold2:
        diff_value = stat_metric - threshold2
        diff_pct = diff_value/threshold2
    else:
        diff_value = 0
        diff_pct = 0

    detect_alg = ((stat_metric < threshold1) or (stat_metric > threshold2))

    if detect_alg:
        detect_msg = '{} 检测数据异常\n观察组 {} 的 {} 值为 {:.2f}，对比指标历史数据的平均值 {} 超出 {} 倍标准差'.format(detect_type, stat_date, obs_metric, stat_metric, ymean, k)
    else:
        detect_msg = '{} 检测数据正常, have a nice day :-)'.format(detect_type)

    detect_info = output_detect_msg(detect_type, detect_comment, detect_alg, diff_value, diff_pct, detect_msg)
    return detect_info


def boxplot_detect(df, dt_col, stat_date, base_date, obs_metric):
    
    detect_type = 'boxplot'
    detect_comment = '箱型图异常值判断'

    stat_metric = df[df[dt_col].astype(str) == stat_date][obs_metric].values[0]
    df[obs_metric] = df[obs_metric].astype('float')
    q1, q3 = df[obs_metric].quantile(.025), df[obs_metric].quantile(.75)
    iqr = q3 - q1
    threshold1, threshold2 = q1-1.5*iqr, q3+1.5*iqr

    if stat_metric < threshold1:
        diff_value = stat_metric - threshold1
        diff_pct = diff_value/threshold1
    elif stat_metric > threshold2:
        diff_value = stat_metric - threshold2
        diff_pct = diff_value/threshold2
    else:
        diff_value = 0
        diff_pct = 0

    detect_alg = ((stat_metric < threshold1) or (stat_metric > threshold2))

    if detect_alg:
        detect_msg = '{} 检测数据异常\n观察组 {} 的 {} 值为 {:.2f}，超出箱型图分布的异常边界值'.format(detect_type, stat_date, obs_metric, stat_metric)
    else:
        detect_msg = '{} 检测数据正常, have a nice day :-)'.format(detect_type)

    detect_info = output_detect_msg(detect_type, detect_comment, detect_alg, diff_value, diff_pct, detect_msg)
    return detect_info


def zscore_detect(df, dt_col, stat_date, base_date, obs_metric):

    detect_type = 'z-score'
    detect_comment = 'z-score 转换'

    stat_metric = df[df[dt_col].astype(str) == stat_date][obs_metric].values[0]
    df[obs_metric] = df[obs_metric].astype('float')

    ymean = np.mean(df[obs_metric])
    ystd = np.std(df[obs_metric])


    diff_value = 0
    diff_pct = 0

    detect_alg = (abs((stat_metric - ymean)/ystd) > 2.2)

    if detect_alg:
        detect_msg = '{} 检测数据异常\n观察组 {} 的 {} 值为 {:.2f}，zscore 的绝对值大于 2.2'.format(detect_type, stat_date, obs_metric, stat_metric)
    else:
        detect_msg = '{} 检测数据正常, have a nice day :-)'.format(detect_type)

    detect_info = output_detect_msg(detect_type, detect_comment, detect_alg, diff_value, diff_pct, detect_msg)
    return detect_info


if __name__ == '__main__':
    input_date = '20180801'
    base_interval = 1
    hist_interval = 5
    mstag = 'test'
    tbl_name = 'demo_onecloud'
    dt_col = 'dt'
    obs_metric = 'click_nums'
    stat_date, base_date, hist_date = date_gen(input_date, base_interval, hist_interval)
    df, df_hist = load_from_mysql(mstag, tbl_name, dt_col, obs_metric, stat_date, base_date, hist_date)
    df['click_type'] = df['click_type'].astype(str)
    obs_metrics, dim_cols = col_type_convert(df)
    
    # detect_info_dod = dod_threshold_detect(df, dt_col, stat_date, base_date, obs_metric)
    # detect_info_ksigma = k_sigma(3, df)
    # detect_info_boxplot = boxplot_detect(df)
    # detect_info_zscore = zscore_detect(df)

    # logging.info([detect_info_dod['detect_result'], detect_info_ksigma['detect_result'], detect_info_boxplot['detect_result'], detect_info_zscore['detect_result']])