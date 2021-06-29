import logging
import pandas as pd
import pymysql
import datetime

logging.basicConfig(level = logging.INFO, format= '%(asctime)s %(message)s', datefmt='%a %d %b %Y %H:%M:%S')

# 日期处理
def date_gen(stat_date, base_interval, hist_interval, is_line):
    stat_date_ = datetime.datetime.strptime(str(stat_date),'%Y%m%d').date() # <class 'datetime.date'>
    base_interval = -abs(base_interval)
    base_date_ = stat_date_ + datetime.timedelta(days=base_interval) # <class 'datetime.date'>
    hist_interval = -abs(hist_interval)
    hist_date_ = stat_date_ + datetime.timedelta(days=hist_interval) # <class 'datetime.date'>

    if is_line:
        stat_date = str(stat_date_) # YYYY-MM-DD
        base_date = datetime.datetime.strftime(base_date_,'%Y-%m-%d') # YYYY-MM-DD
        hist_date = datetime.datetime.strftime(hist_date_,'%Y-%m-%d')
    else:
        stat_date = stat_date
        base_date = datetime.datetime.strftime(base_date_,'%Y%m%d') # YYYYMMDD
        hist_date = datetime.datetime.strftime(hist_date_,'%Y%m%d') # YYYYMMDD
     # YYYY-MM-DD

    logging.info('统计日期:{}，基线日期:{}，历史起始日:{}'.format(stat_date, base_date, hist_date,))
    return stat_date, base_date, hist_date


# 读取 csv 文件
def load_from_csv(path):
    df = pd.read_csv(path)
    return df

# 连接
def load_from_mysql(mstag, sql):
    if mstag == 'test':
        connect_test = pymysql.connect(
            host = 'localhost',
            user = 'root',
            passwd = '020Ac0769',
            db = 'tmp',
            charset = 'utf8'
        )
        cursor = connect_test.cursor()

    logging.info('查询语句: {}'.format(sql))
    cursor.execute(sql)
    query_result = cursor.fetchall()
    query_columns = cursor.description

    cols = []
    for i in range(len(query_columns)):
        cols.append(query_columns[i][0])
    
    df = pd.DataFrame(list(query_result), columns = cols)
    
    # db.close()
    return df

# 读取 MySQL 的数据
# def load_from_mysql(mstag, tbl_name, dt_col, metric_col, stat_date, base_date, hist_date):

    """
    传参构建 SQL 查询语句
    mstag: MySQL 连接串标识
    tbl: 表名
    dt_col: 日期字段
    stat_date: 观察日期
    base_date: 基线日期
    hist_date: 历史日期
    metric_col: 观察指标字段
    """
    # 必须加入日期的字段
    # date_cols = ['fdate', 'date', 'dt', 'dayno', 'fhour', 'fdate']
    # if any(c in stat_date for c in date_cols) and any(c in base_date for c in date_cols):
    #     date_col = [x for i, x in enumerate(date_cols) if x in obs_cond][0] # output: 'dt'
    
    # 获取观察日期和基期的明细数据
    # sql1= 'select * from {} where 1 = 1 and dt between "{}" and "{}"'.format(tbl_name, hist_date, stat_date)
    # df = msyql_to_df(mstag, sql1)

    # 获取观察日期前
    # sql2 = 'select {}, sum({}) {} from {} where dt between "{}" and "{}" group by dt;'.format(dt_col, metric_col, metric_col, tbl_name, hist_date, stat_date)
    # df_hist_metrics = msyql_to_df(mstag, sql2)

    # return df, df_hist_metrics

# def get_obs_df(df, dt_col, stat_date, base_date, obs_metric):
#     df = df.query('{} in ["{}", "{}"]'.format(dt_col, stat_date, base_date))
#     logging.info(df)
#     df_hist = df.groupby(by = dt_col)[obs_metric].sum().reset_index()
#     logging.info(df_hist)
#     return df, df_hist

def col_type_convert(df):

    # 将日期字段强制转换为字符串
    for col in df.columns:
        if col in ['dayno', 'dt', 'date', 'hour', 'fdate', 'fhour']:
            df[col] = df[col].astype(str)
    
    metrics = df._get_numeric_data().columns.tolist()
    dims = df.loc[:, ~df.columns.isin(metrics)].columns.tolist()
    logging.info('指标列: {}'.format(metrics))
    logging.info('维度列: {}'.format(dims))
    return metrics, dims



if __name__ == '__main__':
    stat_date, base_date, hist_date = date_gen('20210421', 7, 7, False)
    logging.info('{},{},{}'.format(stat_date, base_date, hist_date))
    # df, df_hist_metrics = load_from_mysql('test', 'demo_onecloud', 'dt', 'click_nums', stat_date, base_date, hist_date)
    # metric_cols, dim_cols = col_type_convert(df)


