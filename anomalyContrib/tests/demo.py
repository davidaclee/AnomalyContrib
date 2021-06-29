# %%
import logging
import pandas as pd
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json

# 设置日志输出等级
logging.basicConfig(level = logging.DEBUG,
                    format= '%(asctime)s %(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S')

# 设置保留的小数位
pd.set_option('display.float_format', lambda x: '%.6f' % x)

# %%
# 读取数据
df = pd.read_csv('/Users/davidaclee/Desktop/data_coding/data_project/anomaly_contribution_analysis/test.csv')
# df = pd.read_csv(r'D:\桌面\Issues\Programs\abnormal_detect\test2.csv')


# %%
def col_type_analsis(df):
    for col in df.columns:
        if col in ['dayno', 'dt', 'date', 'hour', 'fdate', 'fhour']:
            df[col] = df[col].astype(str)
    
    metric_cols = df._get_numeric_data().columns.tolist()
    print(metric_cols)
    dim_cols = df.loc[:, ~df.columns.isin(dim_cols)].columns.tolist()
    print(dim_cols)
    
    return metric_cols, dim_cols


metric_cols, dim_cols = col_type_analsis(df)
print(metric_cols)

# 如何判断Pandas/NumPy中的列/变量是否为数字?
# https://pyerror.com/detail/4006/


# dim_cols = 

# %%
stat_date = '20210421'
base_date = '20210414'

# 指标列表
# 考虑用字典维护
abs_metrics = ['req_nums', 'req_fill_nums', 'expose_nums', 'acc_nums', 'acc_cost']
cr_metrics = ['req_nums', 'req_fill_rate', 'expose_rate', 'ctr', 'ppc']
core_metric = 'acc_cost'
dim_cols = ['ad_show_pos', 'ad_owner_id']

# 函数：get_rel_metrics(df)
# 用途：计算转化率类指标
# 参数：dataframe
def get_rel_metrics(df):
    df['req_fill_rate'] = df['req_fill_nums']/df['req_nums']
    df['expose_rate'] = df['expose_nums']/df['req_fill_nums']
    df['ctr'] = df['acc_nums']/df['expose_nums']
    df['ppc'] = df['acc_cost']/df['acc_nums']
    return df

# %%
# 异常检测
def threshold_detect(a, b):
    detect_alg = abs(float(a)/float(b) - 1) > 0.1
    diff_type = lambda x: '上涨' if x > 0 else '下降'
    detect_msg = {
        'detect_type': 'threshold',
        'detect_result': detect_alg,
        'diff_value': float(a) - float(b),
        'diff_pct': (float(a) - float(b))/float(b),
        'diff_type': diff_type(float(a) - float(b))
    }
    return detect_msg

# alarm_condition = {
#     'threshold': threshold_detect(a, b)
# }

# if alarm_condition['threshold']:
#     print('数据异常')

# 函数: anomaly_detect(df, dim, metric, stat_date, base_date)
# 用途：数据异常诊断及诊断结果输出
# 参数：dataframe，聚合维度，统计指标，观察日期，实验日期
def anomaly_detect(df, dim, metric, stat_date, base_date):
    df_metric_groupby = df.groupby(by = dim)[metric].sum().reset_index()
    df_stat = df[df[dim] == int(stat_date)].groupby(by = [dim])[metric].sum().values[0]
    df_base = df[df[dim] == int(base_date)].groupby(by = [dim])[metric].sum().values[0]
    msg = threshold_detect(df_stat, df_base)
    # print(msg)
    if msg['detect_result']:
        print('诊断结果:数据异常')
        print('诊断方法:{}'.format(msg['detect_type']))
        print('观察组 {} 的 {} 值为 {:.2f}，对比基期 {} {} {:.2%}'.format(stat_date, metric, df_stat, base_date, msg['diff_type'], msg['diff_pct']))
    else:
        print('数据正常, have a nice day :-)')
    return df_metric_groupby


df_metric_groupby = anomaly_detect(df, 'dayno', core_metric, stat_date, base_date)
df_metric_groupby

# %%
def metric_detect(df, dim):

    # 指标
    df_metric_groupby = df.groupby(by = dim)[core_metric].sum().reset_index()
    df_stat = get_rel_metrics(df[df[dim]==int(stat_date)].groupby(by=[dim])[abs_metrics].sum())
    df_base = get_rel_metrics(df[df[dim]==int(base_date)].groupby(by=[dim])[abs_metrics].sum())

    df_combine = pd.concat([df_stat, df_base])[cr_metrics]
    print('df_combine', df_combine)
    # 最大的影响因素 - 因子乘法
    effect_core_metric = 'effect_' + core_metric
    df_combine_t = df_combine.T.reset_index()
    df_combine_t = df_combine_t.rename(columns = {'index': 'metric'})
    df_combine_t['diff_pct'] = df_combine_t[int(stat_date)]/df_combine_t[int(base_date)]
    df_combine_t['diff_value'] = df_combine_t[int(stat_date)]-df_combine_t[int(base_date)]
    df_combine_t[effect_core_metric] = 0

    # 根据因子乘法获得影响系数最大的指标
    df_combine_t1 = df_combine_t.sort_values(by = 'diff_pct', ascending = False)
    causal_metric = df_combine_t1['metric'].values[0]
    # print(df_combine_t1[['metric', 'diff_pct']].head(5))
    print('根据指标拆解, 影响观察指标波动的因素如下:{}, 其中'.format(cr_metrics))
    
    factor_list = []
    for factor in cr_metrics:
        factor_list.append(factor)
        if len(factor_list) >= 2:
            pre_factor = factor_list[-2] # 取前一个维度
        else:
            pre_factor = int(base_date)
        df_combine_t[factor] = df_combine_t.apply(lambda x: x[int(stat_date)] if x['metric'] in factor_list else x[int(base_date)], axis = 1)
        df_combine_t.loc[lambda x: x['metric'] == factor, effect_core_metric] = df_combine_t[factor].product() - df_combine_t[pre_factor].product()

        print('- {} 的影响系数为: {:.4f}, 影响金额为: {}'.format( \
                factor,
                df_combine_t1[df_combine_t1['metric']==factor]['diff_pct'].values[0],
                df_combine_t[factor].product() - df_combine_t[pre_factor].product()
            ) \
        )
    print('\n经对比, {} 指标的影响最大！'.format(causal_metric))
    causal_metric_delta = df_combine_t[df_combine_t['metric'] == causal_metric]['diff_value'].values[0]
    causal_metric_stat = df_combine_t[df_combine_t['metric'] == causal_metric][int(stat_date)].values[0]
    causal_metric_base = df_combine_t[df_combine_t['metric'] == causal_metric][int(base_date)].values[0]

    # return df_combine_t, causal_metric, causal_metric_delta, causal_metric_stat, causal_metric_base

    
    # df_stat_metric = df_metric_groupby[df_metric_groupby['dayno']==int(stat_date)].rename(columns = {'dayno': 'metric', core_metric:effect_core_metric})
    df_stat_metric = df_metric_groupby[df_metric_groupby['dayno']==int(stat_date)]
    df_stat_metric.columns = ['metric', effect_core_metric]
    # df_base_metric = df_metric_groupby[df_metric_groupby['dayno']==int(base_date)].rename(columns = {'dayno': 'metric', core_metric:effect_core_metric})
    df_base_metric = df_metric_groupby[df_metric_groupby['dayno']==int(base_date)]
    df_base_metric.columns = ['metric', effect_core_metric]

    df_metric_wf = pd.concat([df_base_metric, df_combine_t[['metric', effect_core_metric]], df_stat_metric])

    plot_waterfall(df_metric_wf)

    ##################################################################
    # 维度拆解

    dim_analysis = []

    for col in dim_cols:
        # print(col)
        causal_metric_deno = 'req_fill_nums'
        causal_metric_deno_total = causal_metric_deno + 'total'
        df_dim_detect = get_rel_metrics(df.groupby(by=['dayno', col]).sum())[[causal_metric_deno, causal_metric]].reset_index()
        df_dim_detect[causal_metric_deno_total] = df_dim_detect.groupby('dayno')[causal_metric_deno].transform('sum')
        df_dim_detect['dim_pct'] = df_dim_detect[causal_metric_deno]/df_dim_detect[causal_metric_deno_total]
        # print(df_dim_detect)

        for dim_value in set(df_dim_detect[col]):
            q1 = df_dim_detect[(df_dim_detect['dayno'] == int(base_date)) & (df_dim_detect[col]==dim_value)][causal_metric].values[0]
            q2 = df_dim_detect[(df_dim_detect['dayno'] == int(stat_date)) & (df_dim_detect[col]==dim_value)][causal_metric].values[0]
            w1 = df_dim_detect[(df_dim_detect['dayno'] == int(base_date)) & (df_dim_detect[col]==dim_value)]['dim_pct'].values[0]
            w2 = df_dim_detect[(df_dim_detect['dayno'] == int(stat_date)) & (df_dim_detect[col]==dim_value)]['dim_pct'].values[0]
            Q1 = causal_metric_base

            between_contrib_value = (q1-Q1)*(w2-w1)
            within_attr_value = (q2-q1)*w2
            # print('between_contrib: {}\nwithin_contrib: {}'.format(between_contrib_value, within_attr_value))

            info = {
                'col': col,
                'dim_value': dim_value,
                'base_metric': q1,
                'stat_metric': q2,
                'base_pct': w1,
                'stat_pct': w2,
                'between_contrib': between_contrib_value,
                'within_contrib': within_attr_value,
                'causal_metric_delta': causal_metric_delta
            }
            dim_analysis.append(info)
    print(info)
    df_dim_analysis = pd.DataFrame(dim_analysis)
    df_dim_analysis['dim_value_count'] = df_dim_analysis.groupby('col')['dim_value'].transform('count')
    df_dim_analysis['between_contrib_rate'] = df_dim_analysis['between_contrib']/(df_dim_analysis['causal_metric_delta']/df_dim_analysis['dim_value_count'])
    df_dim_analysis['within_contrib_rate'] = df_dim_analysis['within_contrib']/(df_dim_analysis['causal_metric_delta']/df_dim_analysis['dim_value_count'])
    # between_contrib_cause = 
    if df_dim_analysis[df_dim_analysis['between_contrib_rate'] > 1.5].shape[0] > 0:
        tmp_df_1 = df_dim_analysis[df_dim_analysis['between_contrib_rate'] > 1.5]
        print('拆解维度后发现，{} 维度下 {} 的占比发生变化，对整体 {} 指标的波动贡献度为 {:%}'.format(tmp_df_1['col'].values[0], tmp_df_1['dim_value'].values[0], causal_metric, causal_metric,  tmp_df_1['between_contrib_rate'].values[0]))

        df_dim_wc = df_dim_analysis[df_dim_analysis['col'] == tmp_df_1['col'].values[0]][['dim_value', 'within_contrib']].rename(columns = {'within_contrib': 'contrib'})
        df_dim_wc['dim_value'] += '_wc'
        df_dim_bc = df_dim_analysis[df_dim_analysis['col'] == tmp_df_1['col'].values[0]][['dim_value', 'between_contrib']].rename(columns = {'between_contrib': 'contrib'})
        df_dim_bc['dim_value'] += '_bc'

        df_combine = df_combine.reset_index()
        df_base_causal_metric = df_combine[df_combine['dayno'] == int(base_date)][['dayno', causal_metric]]
        df_base_causal_metric.columns = ['dim_value', 'contrib']
        df_stat_causal_metric = df_combine[df_combine['dayno'] == int(stat_date)][['dayno', causal_metric]]
        df_stat_causal_metric.columns = ['dim_value', 'contrib']

    if df_dim_analysis[df_dim_analysis['within_contrib_rate'] > 1.5].shape[0] > 0:
        tmp_df_2 = df_dim_analysis[df_dim_analysis['within_contrib_rate'] > 1.5]
        print('拆解维度后发现，{} 维度下 {} 的 {} 指标出现明显下降，对整体 {} 指标的波动贡献度为 {:2%}'.format(tmp_df_2['col'].values[0], tmp_df_2['dim_value'].values[0], causal_metric, causal_metric,  tmp_df_2['within_contrib_rate'].values[0]))
        
        df_dim_wc = df_dim_analysis[df_dim_analysis['col'] == tmp_df_2['col'].values[0]][['dim_value', 'within_contrib']].rename(columns = {'within_contrib': 'contrib'})
        df_dim_wc['dim_value'] += '_wc'
        df_dim_bc = df_dim_analysis[df_dim_analysis['col'] == 'ad_show_pos'][['dim_value', 'between_contrib']].rename(columns = {'between_contrib': 'contrib'})
        df_dim_bc['dim_value'] += '_bc'

        df_combine = df_combine.reset_index()
        df_base_causal_metric = df_combine[df_combine['dayno'] == int(base_date)][['dayno', causal_metric]]
        df_base_causal_metric.columns = ['dim_value', 'contrib']
        df_stat_causal_metric = df_combine[df_combine['dayno'] == int(stat_date)][['dayno', causal_metric]]
        df_stat_causal_metric.columns = ['dim_value', 'contrib']

        df_dim_wf = pd.concat([df_base_causal_metric, df_dim_wc, df_dim_bc, df_stat_causal_metric])
        plot_waterfall(df_dim_wf)
    return df_combine, df_combine_t, causal_metric, causal_metric_delta, df_dim_analysis


df_combine, df_combine_t, causal_metric, causal_metric_delta, df_dim_analysis = metric_detect(df, 'dayno')


 # %%
def draw_waterfall(df):
    
    print(trans)
    print(trans['metric'].tolist())

    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ['absolute', 'relative', 'relative', 'relative', 'relative', 'relative', 'total'],
        x = trans['metric'].tolist(),
        textposition = 'outside',
        text = trans['effect_cost'].round(2).astype('str').tolist(),
        y = trans['effect_cost'].tolist(),
        connector = {'line':{'color':'rgb(63, 63, 63)'}},
    ))

    fig.update_layout(
            title = '各因子影响金额瀑布图(经对比, {} 指标的影响最大！)'.format(causal_metric),
            showlegend = False,
            waterfallgap = 0.3
    )

    fig.show()



# df_dim_analysis[df_dim_analysis['col'] == 'ad_show_pos'][['dim_value', 'within_contrib_rate']]
# %%
# def dim_detect():
#     for col in dim_cols:
#         df_dim = get_rel_metrics(df.groupby(by=['dayno', col]).sum())[causal_metric].reset_index()
#         for dim_value in set(df_dim[col]):
#             df_dim_stat = df_dim[(df_dim['dayno']==int(stat_date)) & (df_dim[col]==dim_value)][causal_metric].values[0]
#             df_dim_base = df_dim[(df_dim['dayno']==int(base_date)) & (df_dim[col]==dim_value)][causal_metric].values[0]
#             dim_value_delta = df_dim_stat - df_dim_base
#             print('{} 维度下 {} 在 {} 指标的波动值为 {}'.format(col, dim_value, causal_metric, dim_value_delta))
# dim_detect()

# # tran2
def plot_waterfall(df):
    measure = ['abosulute']
    for i in range(df.shape[0]-2):
        measure.append('relative')
    measure.append('total')
    print(measure)
    
    dim_col = df.columns[0]
    value_col = df.columns[1]
    df[dim_col] = df[dim_col].astype(str)
    print(df)
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = measure,
        x = df[dim_col].tolist(),
        textposition = 'outside',
        text = df[value_col].round(4).astype('str').tolist(),
        y = df[value_col].tolist(),
        connector = {'line':{'color':'rgb(63, 63, 63)'}},
    ))

    fig.update_layout(
            title = '维度拆解'.format(causal_metric, ),
            showlegend = False,
            waterfallgap = 0.5
    )

    fig.show()

# %%
# def giniCoef(wealths):
#     cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
#     sum_wealths = cum_wealths[-1]
#     xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
#     yarray = cum_wealths / sum_wealths
#     B = np.trapz(yarray, x=xarray)
#     A = 0.5 - B
#     return A / (A+B)

# wealths = [100, 40, 20]

# giniCoef(wealths)


