import pandas as pd
import logging

# 函数：get_comp_metric()
# 用途：计算转化率类指标
# 参数：dataframe
def cal_comp_metric(df):
    df['req_fill_rate'] = df['req_fill_nums']/df['req_nums']
    df['expose_rate'] = df['expose_nums']/df['req_fill_nums']
    df['ctr'] = df['acc_nums']/df['expose_nums']
    df['ppc'] = df['acc_cost']/df['acc_nums']
    return df

def get_factor_form(df, dt_col, stat_date, base_date, obs_metric, stat_metric, comp_metric):
    df_factor_form = cal_comp_metric(df.query('{} in ["{}", "{}"]'.format(dt_col, stat_date, base_date)).groupby(by = dt_col)[stat_metric].sum())[comp_metric]
    # %%
    df_factor_form_t = df_factor_form.T.reset_index()
    df_factor_form_t = df_factor_form_t.rename(columns = {'index': 'metric'})
    impact_metric = 'impact_' + obs_metric
    df_factor_form_t[impact_metric] = 0
    df_factor_form_t['diff_pct'] = df_factor_form_t[stat_date]/df_factor_form_t[base_date]
    df_factor_form_t['diff_value'] = df_factor_form_t[stat_date]-df_factor_form_t[base_date]

    factor_list = []
    for factor in comp_metric:
        factor_list.append(factor)
        if len(factor_list) >= 2:
            pre_factor = factor_list[-2] # 取前一个维度
            logging.info(pre_factor)
        else:
            pre_factor = int(base_date)
            logging.info(pre_factor)

        df_factor_form_t[factor] = df_factor_form_t.apply(lambda x: x[int(stat_date)] if x['metric'] in factor_list else x[int(base_date)], axis = 1)
        df_factor_form_t.loc[lambda x: x['metric'] == factor, impact_metric] = df_factor_form_t[factor].product() - df_factor_form_t[pre_factor].product()
        print(df_factor_form_t)
        # print('- {} 的影响系数为: {:.4f}, 影响金额为: {}'.format( \
        #         factor,
        #         df_factor_form_t[df_factor_form_t['metric']==factor]['diff_pct'].values[0],
        #         df_factor_form_t[factor].product() - df_factor_form_t[pre_factor].product()
        #     ) \
        # )
    # print('\n经对比, {} 指标的影响最大！'.format(causal_metric))
    # causal_metric_delta = df_combine_t[df_combine_t['metric'] == causal_metric]['diff_value'].values[0]
    # causal_metric_stat = df_combine_t[df_combine_t['metric'] == causal_metric][int(stat_date)].values[0]
    # causal_metric_base = df_combine_t[df_combine_t['metric'] == causal_metric][int(base_date)].values[0]

    # # return df_combine_t, causal_metric, causal_metric_delta, causal_metric_stat, causal_metric_base

    
    # # df_stat_metric = df_metric_groupby[df_metric_groupby['dayno']==int(stat_date)].rename(columns = {'dayno': 'metric', core_metric:effect_core_metric})
    # df_stat_metric = df_metric_groupby[df_metric_groupby['dayno']==int(stat_date)]
    # df_stat_metric.columns = ['metric', effect_core_metric]
    # # df_base_metric = df_metric_groupby[df_metric_groupby['dayno']==int(base_date)].rename(columns = {'dayno': 'metric', core_metric:effect_core_metric})
    # df_base_metric = df_metric_groupby[df_metric_groupby['dayno']==int(base_date)]
    # df_base_metric.columns = ['metric', effect_core_metric]

    # df_metric_wf = pd.concat([df_base_metric, df_combine_t[['metric', effect_core_metric]], df_stat_metric])

    
    return df_factor_form_t

if __name__ == '__main__':
    pass