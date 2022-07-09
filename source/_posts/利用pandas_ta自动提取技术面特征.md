---
title: 利用pandas_ta自动提取技术面特征
date: 2022-07-09 18:00:00
tags: 
    - talib
    - pandas_ta
    - 技术分析
categories:
    - 量化
---

> TA-Lib是一个技术分析库，涵盖了150多种股票、期货交易软件中常用的技术分析指标，如MACD、RSI、KDJ、动量指标、布林带等等，而`pandas-ta`则是一个基于pandas和ta-lib的高级技术分析工具，具有​​130多个指标和实用功能以及60多个TA-Lib中包含的蜡烛模式。本章节记录如何利用`pandas-ta`快速提取技术面特征。

## 安装

安装`pandas-ta`本身非常简单，直接pip一下就可以，如果想用ta-lib的一些特性，则还需要安装`ta-lib`本身
- Anaconda：使用Anaconda集成环境，可直接`conda install -c conda-forge ta-lib`
- Windows: 直接在https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib 中下载离线包安装即可
- Linux: 需要编译安装C++原始环境库(http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz) ，然后pip安装`pip install TA-Lib`

pandas-ta提供的主要函数有：
- 蜡烛图形态：Candles ，基于K线图的形态识别，如三只乌鸦等
- 周期特征：Cycles，如正弦波拟合等
- 动量特征：Momentum，如KDJ、RSI等
- 覆盖特征：Overlap，主要包含各种移动平均线系列，如EMA指数平滑
- 回报特征：Performance ，如百分比回报、log回报等
- 统计特征：Statistics，如熵、中位数、分位数、标准差等
- 趋势特征：Trend， 如阿隆指数等
- 波动率特征：Volatility，如布林带等 
- 成交量特征：Volume，如资金流指数等
- 其他特征：如神奇九转(td_seq)等

## 准备K线数据

这里主要提取1min和1d两种粒度数据中的特征，日内1分钟数据未除权，日间1天数据使用后复权，保持序列的连贯性。

### 读取日内K线数据(未复权)
这里以聚宽数据为例，读取指定时间区间中1min数据，并生成对应的日内收益标签，详细功能见代码中的注释。

主要的收益计算为日内指标，分为有两类：
- 日内剩余时间段里的平均收益百分比
- 日内随后5min、10min的平均收益

```python
# 日内：聚宽
def load_jqdata_kline(code='510050.SH', start_date='2021-11-01', end_date='2022-01-01'):
    # 代码转换
    if not code.endswith(('.XSHG', '.XSHE')):
        ticker, exchange  = code.split('.')
        code = ticker + '.' + {'SZ': 'XSHE', 'BJ': 'XSHE', 'SH': 'XSHG'}.get(exchange.upper())
    # 提取stock_kline中的1min数据
    query = "select * from jqdata.stock_kline where code='{}' and volume !=0.0  and trade_day >='{}' and trade_day <'{}' and toDayOfWeek(trade_day) < 6".format(code, start_date, end_date)
    data = client.execute(query)

    # 提取stock_dayline中的1d数据,主要是复权因子
    day_query = "select trade_day, pre_close, factor from jqdata.stock_dayline where code='{}' and volume !=0.0  and trade_day >='{}' and trade_day <'{}' and toDayOfWeek(trade_day) < 6".format(code, start_date, end_date)
    day_data = client.execute(day_query)
    # 获取复权后的昨收数据
    pre_close = {i[0]: np.round(i[1] / i[2] , 3) for i in  day_data}

    df = pd.DataFrame(data, columns=['code', 'dtype', 'trade_day', 'trade_time', 'open', 'close', 'low', 'high', 'volume', 'money', 'avg'])
    df = df.sort_values(by='trade_time').drop_duplicates(subset=['trade_time'], keep='first').reset_index(drop=True)
    # 补充昨收数据
    df['pre_close'] = df['trade_day'].apply(lambda x: pre_close.get(x, 0))
    df['trade_day'] = df['trade_day'].astype(str)

    # 当天收盘相对于昨收的百分比
    df['daily_percent'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['minutes_of_day'] = df.trade_time.dt.hour * 60 + df.trade_time.dt.minute

    # 过滤尾盘集合竞价数据    
    df = df[~df.minutes_of_day.isin([898, 899])]

    # 此刻买进，日内剩余时间段内的收益25%分位数
    df['label_d0_25'] = ((df.groupby(['trade_day'])['avg'].transform(lambda row: row[::-1].shift(1).expanding(min_periods=1).quantile(0.25)[::-1]) - df['avg']) * 100).div(df['avg'])
    # 此刻买进，日内剩余时间段内的收益50%分位数
    df['label_d0_50'] = ((df.groupby(['trade_day'])['avg'].transform(lambda row: row[::-1].shift(1).expanding(min_periods=1).quantile(0.5)[::-1]) - df['avg']) * 100).div(df['avg'])
    # 此刻买进，日内剩余时间段内的收益75%分位数
    df['label_d0_75'] = ((df.groupby(['trade_day'])['avg'].transform(lambda row: row[::-1].shift(1).expanding(min_periods=1).quantile(0.75)[::-1]) - df['avg']) * 100).div(df['avg'])

    # 此刻买进，日内5分钟时间段内的收益50%分位具体值
    df['label_rolling_5_50_bps'] = (df.groupby(['trade_day'])['avg'].transform(lambda row: row[::-1].shift(1).rolling(5, min_periods=1).quantile(0.5)[::-1]) - df['avg']) * 100
    # 此刻买进，日内5分钟时间段内的收益50%分位数
    df['label_rolling_5_50'] = df['label_rolling_5_50_bps'].div(df['avg'])
    # 此刻买进，日内10分钟时间段内的收益50%分位具体值
    df['label_rolling_10_50_bps'] = df.groupby(['trade_day'])['avg'].transform(lambda row: row[::-1].shift(1).rolling(10, min_periods=1).quantile(0.5)[::-1]) - df['avg']
    # 此刻买进，日内10分钟时间段内的收益50%分位数
    df['label_rolling_10_50'] = df['label_rolling_10_50_bps'].div(df['avg'])

    return df.reset_index(drop=True)

```

### 读取日间K线数据(后复权)

日线有两个，一个是聚宽数据，主要针对国内市场，还有一个是华盛通数据，针对香港和美国市场，都从clickhouse数据库中读取。

```python
# 日线: 华盛通
def load_hstong_dayline(market='cn', code='300369.SZ', start_date='2018-11-01', end_date='2022-01-01'):
    query = "select code, data_date, price_open, price_close, price_low, price_high, price_last, volume, deal  from hstong.dayline_hfq_{} where code='{}' and volume !=0.0  and data_date >='{}' and data_date <='{}' and toDayOfWeek(data_date) < 6".format(market, code, start_date, end_date)
    data = client.execute(query)
    df = pd.DataFrame(data, columns=['code', 'trade_day', 'open', 'close', 'low', 'high', 'pre_close', 'volume', 'money'])
    df = df.sort_values(by='trade_day').drop_duplicates(subset=['trade_day'], keep='first').reset_index(drop=True)
    return df

# 日线：聚宽
def load_jqdata_dayline(code='510050.SH', start_date='2018-11-01', end_date='2022-01-01'):
    if not code.endswith(('.XSHG', '.XSHE')):
        ticker, exchange  = code.split('.')
        code = ticker + '.' + {'SZ': 'XSHE', 'BJ': 'XSHE', 'SH': 'XSHG'}.get(exchange.upper())
    query = "select code, trade_day, open, close, low, high, pre_close, volume, money, factor  from jqdata.stock_dayline where code='{}' and volume !=0.0  and trade_day >='{}' and trade_day <'{}' and toDayOfWeek(trade_day) < 6".format(code, start_date, end_date)
    data = client.execute(query)
    df = pd.DataFrame(data, columns=['code', 'trade_day', 'open', 'close', 'low', 'high', 'pre_close', 'volume', 'money', 'factor'])
    df = df.sort_values(by='trade_day').drop_duplicates(subset=['trade_day'], keep='first').reset_index(drop=True)
    df['trade_day'] = df['trade_day'].astype(str)
    return df
```

我们以茅台(600519.SH)为例,读取'2020-01-01'至'2022-07-07'间的数据：
```
kline_df = load_jqdata_kline(code='600519.SH', start_date='2020-01-01', end_date='2022-07-07')
print(kline_df.iloc[-5:])
```

| code        | dtype   | trade_day   | trade_time                |    open |   close |     low |    high |   volume |       money |     avg |   pre_close |   daily_percent |   minutes_of_day |   label_d0_25 |   label_d0_50 |   label_d0_75 |   label_rolling_5_50_bps |   label_rolling_5_50 |   label_rolling_10_50_bps |   label_rolling_10_50 |
|:------------|:--------|:------------|:--------------------------|--------:|--------:|--------:|--------:|---------:|------------:|--------:|------------:|----------------:|-----------------:|--------------:|--------------:|--------------:|-------------------------:|---------------------:|--------------------------:|----------------------:|
| 600519.XSHG | STOCK   | 2022-07-07  | 2022-07-07 14:54:00+08:00 | 1994.86 | 1993.66 | 1993.66 | 1994.86 |    10500 | 2.09366e+07 | 1993.99 |        2002 |     -0.00416583 |              894 |    -0.0850054 |    -0.0308427 |    -0.0132899 |                    -61.5 |           -0.0308427 |                    -0.615 |          -0.000308427 |
| 600519.XSHG | STOCK   | 2022-07-07  | 2022-07-07 14:55:00+08:00 | 1993.58 | 1994.04 | 1993.57 | 1994.5  |     8900 | 1.77452e+07 | 1993.83 |        2002 |     -0.00397602 |              895 |    -0.115356  |    -0.0386191 |    -0.0228204 |                    -77   |           -0.0386191 |                    -0.77  |          -0.000386191 |
| 600519.XSHG | STOCK   | 2022-07-07  | 2022-07-07 14:56:00+08:00 | 1993.99 | 1993.68 | 1993.51 | 1993.99 |    18700 | 3.72816e+07 | 1993.69 |        2002 |     -0.00415584 |              896 |    -0.146713  |    -0.108342  |    -0.0699708 |                   -216   |           -0.108342  |                    -2.16  |          -0.00108342  |
| 600519.XSHG | STOCK   | 2022-07-07  | 2022-07-07 14:57:00+08:00 | 1993.29 | 1993    | 1993    | 1993.29 |    15900 | 3.16902e+07 | 1993.06 |        2002 |     -0.0044955  |              897 |    -0.153533  |    -0.153533  |    -0.153533  |                   -306   |           -0.153533  |                    -3.06  |          -0.00153533  |
| 600519.XSHG | STOCK   | 2022-07-07  | 2022-07-07 15:00:00+08:00 | 1993.01 | 1990    | 1990    | 1993.01 |    47800 | 9.51379e+07 | 1990    |        2002 |     -0.00599401 |              900 |   nan         |   nan         |   nan         |                    nan   |          nan         |                   nan     |         nan           |

## 构建日内特征(分钟级)

读取K线数据后，便进行最重要的特征提取工作，同时生成对应的标签

```python
def make_tech_feature_intraday_jqdata(kline_df, index_col='trade_time'):
    df_grouped = kline_df.set_index(index_col).groupby('trade_day', as_index=False)

    rsi_5 = (df_grouped.apply(lambda row: row.ta.rsi(length=5) / 100)).fillna(0.5).to_frame().reset_index(drop=True)
    cmo_3 =  (df_grouped.apply(lambda row: row.ta.cmo(length=3) / 100)).fillna(0.0).to_frame().reset_index(drop=True)
    natr_3 =  (df_grouped.apply(lambda row: row.ta.natr(length=3))).fillna(0.0).to_frame().reset_index(drop=True)
    rvi_5 =  (df_grouped.apply(lambda row: row.ta.rvi(length=5) / 100)).fillna(0.5).to_frame().reset_index(drop=True)
    try:
        kdj_9_3 = (df_grouped.apply(lambda row: row.ta.kdj(9, 3) / 100)).fillna(0.5).reset_index(drop=True)
    except:
        kdj_9_3 = (df_grouped.apply(lambda row: row.ta.kdj(min(9, len(row)), 3).T.reset_index(drop=True).T / 100)).fillna(0.5).reset_index(drop=True)
        kdj_9_3.columns = ['K_9_3', 'D_9_3', 'J_9_3']
    adx_5 = (df_grouped.apply(lambda row: row.ta.adx(length=5) / 100)).clip(-10, 10).fillna(0.0).reset_index(drop=True)
    try:
        mfi_5 = (df_grouped.apply(lambda row: row.ta.mfi(length=5) / 100)).clip(-10, 10).fillna(0.5).to_frame().reset_index(drop=True)
    except:
        mfi_5 = (df_grouped.apply(lambda row: row.ta.mfi(length=min(5, len(row))).rename('MFI_5') / 100)).clip(-10, 10).fillna(0.5).to_frame().reset_index(drop=True)
    pvi_5 = (df_grouped.apply(lambda row: row.ta.pvi(length=5) / 1000)).clip(-10, 10).fillna(1).to_frame().reset_index(drop=True)
    nvi_5 = (df_grouped.apply(lambda row: row.ta.nvi(length=5) / 1000)).clip(-10, 10).fillna(1).to_frame().reset_index(drop=True)
    willr_5 = (df_grouped.apply(lambda row: row.ta.willr(length=5) / 100)).clip(-1, 1).fillna(-0.5).to_frame().reset_index(drop=True)
    try:
        willr_10 = (df_grouped.apply(lambda row: row.ta.willr(length=10) / 100)).clip(-1, 1).fillna(-0.5).to_frame().reset_index(drop=True)
    except:
        willr_10 = (df_grouped.apply(lambda row: row.ta.willr(length=min(10, len(row))).rename('WILLR_10') / 100)).clip(-1, 1).fillna(-0.5).to_frame().reset_index(drop=True)
    try:
        cmf_10 = (df_grouped.apply(lambda row: row.ta.cmf(length=10) )).clip(-10, 10).fillna(0.0).to_frame().reset_index(drop=True)
    except:
        cmf_10 = (df_grouped.apply(lambda row: row.ta.cmf(length=min(10, len(row))).rename('CMF_10') )).clip(-10, 10).fillna(0.0).to_frame().reset_index(drop=True)
    dpo_5 = (df_grouped.apply(lambda row: row.ta.dpo(length=5, lookahead=False) * 10 )).clip(-10, 10).fillna(0.0).to_frame().reset_index(drop=True)
    log_return_5 = (df_grouped.apply(lambda row: row.ta.log_return(length=5) * 50 )).clip(-10, 10).fillna(0.0).to_frame().reset_index(drop=True) 
    zscore_5 = (df_grouped.apply(lambda row: row.ta.zscore(length=5) )).clip(-10, 10).fillna(0.0).to_frame().reset_index(drop=True) 
    pct_change_3 = (df_grouped.apply(lambda row: row['avg'].pct_change(periods=3).fillna(0) * 50 / 3)).clip(-10, 10).to_frame().rename(columns={'avg': 'pct_change_3'}).reset_index(drop=True) 
    pct_change_6 = (df_grouped.apply(lambda row: row['avg'].pct_change(periods=6).fillna(0) * 50 / 6)).clip(-10, 10).to_frame().rename(columns={'avg': 'pct_change_6'}).reset_index(drop=True) 
    
    rolling_std_5 = (df_grouped.apply(lambda row: row['avg'].rolling(5, min_periods=1).std(ddof=0))).clip(-10, 10).to_frame().rename(columns={'avg': 'rolling_std_5'}).reset_index(drop=True)
    rolling_money_5 = (df_grouped.apply(lambda row: np.log1p(row['money'].rolling(5, min_periods=1).mean()) - np.log1p(row['money']) ) / 10.0).clip(-10, 10).to_frame().rename(columns={'money': 'rolling_money_5'}).reset_index(drop=True)
    rolling_money_6_3 = (df_grouped.apply(lambda row: np.log1p(row['money'].rolling(6, min_periods=1).mean()) - np.log1p(row['money'].rolling(3, min_periods=1).mean()) ) / 10.0 ).clip(-10, 10).to_frame().rename(columns={'money': 'rolling_money_6_3'}).reset_index(drop=True)
    rolling_volume_5 = (df_grouped.apply(lambda row: np.log1p(row['volume'].rolling(5, min_periods=1).mean()) - np.log1p(row['volume']) )/ 10.0).clip(-10, 10).to_frame().rename(columns={'volume': 'rolling_volume_5'}).reset_index(drop=True)
    rolling_volume_6_3 = (df_grouped.apply(lambda row: np.log1p(row['volume'].rolling(6, min_periods=1).mean()) - np.log1p(row['volume'].rolling(3, min_periods=1).mean()) )/ 10.0).clip(-10, 10).to_frame().rename(columns={'volume': 'rolling_volume_6_3'}).reset_index(drop=True)
    pct_volatility = (df_grouped.apply(lambda row: (row['high'] - row['low']) * 50 / row['avg'])).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'pct_volatility'}).reset_index(drop=True) 
    rolling_pct_volatility_5 = (df_grouped.apply(lambda row: (row['high'].rolling(5, min_periods=1).max() - row['low'].rolling(5, min_periods=1).min())* 20 / row['avg'])).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'rolling_pct_volatility_5'}).reset_index(drop=True) 
    rolling_pct_volatility_10 = (df_grouped.apply(lambda row: (row['high'].rolling(10, min_periods=1).max() - row['low'].rolling(10, min_periods=1).min()) * 20 / row['avg'])).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'rolling_pct_volatility_10'}).reset_index(drop=True) 
    pct_vwap_low = (df_grouped.apply(lambda row: (row.ta.vwap() - row['low']) * 50 / row['avg'])).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'pct_vwap_low'}).reset_index(drop=True) 
    pct_vwap_high = (df_grouped.apply(lambda row: (row['high'] - row.ta.vwap()) * 50 / row['avg'])).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'pct_vwap_high'}).reset_index(drop=True) 
    
    feature = pd.concat([rsi_5, cmo_3, natr_3, rvi_5, kdj_9_3,  adx_5, mfi_5, pvi_5, nvi_5, willr_5, willr_10, cmf_10, dpo_5, log_return_5, zscore_5, pct_change_3, pct_change_6, rolling_std_5, rolling_money_5, rolling_money_6_3, rolling_volume_5, rolling_volume_6_3, pct_volatility, rolling_pct_volatility_5, rolling_pct_volatility_10, pct_vwap_low, pct_vwap_high, kline_df[['daily_percent']] * 50], axis=1) 
    feature.columns = ['intraday_%s' % i for i in feature.columns]

    index = kline_df[index_col].apply(lambda x: int(x.timestamp()))
    
    introday_data = pd.concat([kline_df[['code', 'trade_day', 'minutes_of_day']], feature], axis=1)
    introday_data.index = index
    

    if 'label_d0_25' in kline_df.columns:
        label_data = kline_df[['code', 'trade_day', 'minutes_of_day', 'label_d0_25', 'label_d0_50', 'label_d0_75',
           'label_rolling_5_50', 'label_rolling_10_50', 'label_rolling_5_50_bps', 'label_rolling_10_50_bps']]
        label_data.index = index
    else:
        label_data = []

    introday_data = introday_data[introday_data.minutes_of_day != 900]    
    label_data = label_data[label_data.minutes_of_day != 900]        
    return introday_data.reset_index(), label_data.reset_index()
```

同样的，我们展示上面茅台数据生成经过pandas-ta提取的特征：
```python
introday_data, label_data = make_tech_feature_intraday_jqdata(kline_df, index_col='trade_time')
print(introday_data.iloc[-5:])
print(label_data.iloc[-5:])

```

- 日内数据特征：

|   trade_time | code        | trade_day   |   minutes_of_day |   intraday_RSI_5 |   intraday_CMO_3 |   intraday_NATR_3 |   intraday_RVI_5 |   intraday_K_9_3 |   intraday_D_9_3 |   intraday_J_9_3 |   intraday_ADX_5 |   intraday_DMP_5 |   intraday_DMN_5 |   intraday_MFI_5 |   intraday_PVI_5 |   intraday_NVI_5 |   intraday_WILLR_5 |   intraday_WILLR_10 |   intraday_CMF_10 |   intraday_DPO_5 |   intraday_LOGRET_5 |   intraday_ZS_5 |   intraday_pct_change_3 |   intraday_pct_change_6 |   intraday_rolling_std_5 |   intraday_rolling_money_5 |   intraday_rolling_money_6_3 |   intraday_rolling_volume_5 |   intraday_rolling_volume_6_3 |   intraday_pct_volatility |   intraday_rolling_pct_volatility_5 |   intraday_rolling_pct_volatility_10 |   intraday_pct_vwap_low |   intraday_pct_vwap_high |   intraday_daily_percent |
|-------------:|:------------|:------------|-----------------:|-----------------:|-----------------:|------------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-------------------:|--------------------:|------------------:|-----------------:|--------------------:|----------------:|------------------------:|------------------------:|-------------------------:|---------------------------:|-----------------------------:|----------------------------:|------------------------------:|--------------------------:|------------------------------------:|-------------------------------------:|------------------------:|-------------------------:|-------------------------:|
|   1657176780 | 600519.XSHG | 2022-07-07  |              893 |         0.4868   |        -0.064103 |         0.077846  |         0.472697 |         0.536684 |         0.41818  |        0.773693  |         0.404289 |         0.170589 |         0.053766 |       0.589231   |         0.999701 |         1.00016  |          -0.377273 |           -0.452    |         0.197077  |              4.1 |           0.0310894 |       -0.381427 |              0.00802214 |             0.00689641  |                 0.613697 |                -0.010953   |                  -0.0104473  |                 -0.011178   |                   -0.0104575  |                0.03508    |                           0.0220503 |                            0.0250571 |               0.0622761 |               -0.0271961 |                -0.178072 |
|   1657176840 | 600519.XSHG | 2022-07-07  |              894 |         0.377828 |        -0.418172 |         0.0721596 |         0.306806 |         0.382032 |         0.406131 |        0.333835  |         0.33549  |         0.144056 |         0.127672 |       0.353767   |         0.999656 |         1.00016  |          -0.971429 |           -0.931624 |        -0.0242737 |            -10   |          -0.0225665 |       -1.63965  |             -0.00860476 |            -0.00167135  |                 0.50677  |                -0.0136592  |                  -0.006727   |                 -0.0136515  |                   -0.00678157 |                0.0300904  |                           0.0210633 |                            0.0234705 |               0.0780107 |               -0.0479203 |                -0.208292 |
|   1657176900 | 600519.XSHG | 2022-07-07  |              895 |         0.428086 |        -0.203659 |         0.0636436 |         0.536757 |         0.336506 |         0.382923 |        0.243674  |         0.269736 |         0.125327 |         0.123655 |       0.199639   |         0.999656 |         1.00009  |          -0.779343 |           -0.754545 |         0.0696795 |             -7.7 |          -0.0391013 |       -0.75092  |             -0.0102754  |             0.000668783 |                 0.638041 |                 0.00777374 |                  -0.00850169 |                  0.00777846 |                   -0.00857828 |                0.0233219  |                           0.0213659 |                            0.0220681 |               0.0799701 |               -0.0566481 |                -0.198801 |
|   1657176960 | 600519.XSHG | 2022-07-07  |              896 |         0.390711 |        -0.344547 |         0.051298  |         0.344622 |         0.25161  |         0.339152 |        0.0765271 |         0.222581 |         0.114705 |         0.122769 |       0.165302   |         0.999611 |         1.00009  |          -0.922374 |           -0.918182 |         0.0739293 |            -10   |          -0.0223156 |       -0.902093 |             -0.0146167  |            -0.00330078  |                 0.709123 |                -0.0476034  |                  -0.0131696  |                 -0.0475763  |                   -0.0131647  |                0.012038   |                           0.0219693 |                            0.0220696 |               0.0807789 |               -0.0687409 |                -0.207792 |
|   1657177020 | 600519.XSHG | 2022-07-07  |              897 |         0.323935 |        -0.563402 |         0.0455835 |         0.243084 |         0.16774  |         0.282015 |       -0.0608087 |         0.242891 |         0.100978 |         0.197831 |       2.9012e-16 |         0.999611 |         0.999951 |          -1        |           -1        |        -0.143719  |            -10   |          -0.0674407 |       -1.39213  |             -0.00777336 |            -0.00818705  |                 0.785147 |                -0.0210747  |                  -0.0162005  |                 -0.0210629  |                   -0.0161831  |                0.00727525 |                           0.027094  |                            0.027094  |               0.0928892 |               -0.0856139 |                -0.224775 |


- 日内标签数据
  
|   trade_time | code        | trade_day   |   minutes_of_day |   label_d0_25 |   label_d0_50 |   label_d0_75 |   label_rolling_5_50 |   label_rolling_10_50 |   label_rolling_5_50_bps |   label_rolling_10_50_bps |
|-------------:|:------------|:------------|-----------------:|--------------:|--------------:|--------------:|---------------------:|----------------------:|-------------------------:|--------------------------:|
|   1657176780 | 600519.XSHG | 2022-07-07  |              893 |    -0.119272  |    -0.0877    |    -0.080684  |           -0.0877    |          -0.000877    |                   -175   |                    -1.75  |
|   1657176840 | 600519.XSHG | 2022-07-07  |              894 |    -0.0850054 |    -0.0308427 |    -0.0132899 |           -0.0308427 |          -0.000308427 |                    -61.5 |                    -0.615 |
|   1657176900 | 600519.XSHG | 2022-07-07  |              895 |    -0.115356  |    -0.0386191 |    -0.0228204 |           -0.0386191 |          -0.000386191 |                    -77   |                    -0.77  |
|   1657176960 | 600519.XSHG | 2022-07-07  |              896 |    -0.146713  |    -0.108342  |    -0.0699708 |           -0.108342  |          -0.00108342  |                   -216   |                    -2.16  |
|   1657177020 | 600519.XSHG | 2022-07-07  |              897 |    -0.153533  |    -0.153533  |    -0.153533  |           -0.153533  |          -0.00153533  |                   -306   |                    -3.06  |


简单计算技术因子与日内收益的相关性：
```python
corr_data = []
for col in introday_data.columns.tolist()[4:]:
    feature = introday_data[col]
    label_d0_50 = feature.corr(label_data['label_d0_50'])
    label_rolling_5_50 = feature.corr(label_data['label_rolling_5_50'])
    label_rolling_10_50 = feature.corr(label_data['label_rolling_10_50'])
    corr_data.append([col, label_d0_50, label_rolling_5_50, label_rolling_10_50])
corr_data = pd.DataFrame(corr_data, columns=['feature', 'd0_50', 'rolling_5_50', 'rolling_10_50'])
print(corr_data)
```

| feature                            |       d0_50 |   rolling_5_50 |   rolling_10_50 |
|:-----------------------------------|------------:|---------------:|----------------:|
| intraday_RSI_5                     |  0.0402828  |     0.134321   |      0.103664   |
| intraday_CMO_3                     |  0.0460717  |     0.163593   |      0.124605   |
| intraday_NATR_3                    | -0.0044438  |     0.0184311  |      0.0159059  |
| intraday_RVI_5                     |  0.0384155  |     0.139606   |      0.106858   |
| intraday_K_9_3                     |  0.0281801  |     0.0653667  |      0.0561566  |
| intraday_D_9_3                     |  0.019375   |     0.0322949  |      0.0315525  |
| intraday_J_9_3                     |  0.0305751  |     0.0899986  |      0.0733925  |
| intraday_ADX_5                     |  0.00178353 |     0.00119938 |     -0.00362684 |
| intraday_DMP_5                     |  0.023113   |     0.0693602  |      0.0517322  |
| intraday_DMN_5                     | -0.0211941  |    -0.0660362  |     -0.0478488  |
| intraday_MFI_5                     |  0.0187832  |     0.0573829  |      0.0454739  |
| intraday_PVI_5                     | -0.00631178 |     0.00495645 |      0.0021088  |
| intraday_NVI_5                     | -0.0138721  |     0.0016334  |     -0.00108481 |
| intraday_WILLR_5                   |  0.0508993  |     0.174907   |      0.135297   |
| intraday_WILLR_10                  |  0.0437711  |     0.133309   |      0.106649   |
| intraday_CMF_10                    |  0.0310067  |     0.0720305  |      0.060817   |
| intraday_DPO_5                     |  0.0322019  |     0.101203   |      0.0789424  |
| intraday_LOGRET_5                  |  0.0282654  |     0.101385   |      0.0758353  |
| intraday_ZS_5                      |  0.048193   |     0.189238   |      0.140897   |
| intraday_pct_change_3              |  0.0111402  |     0.0316397  |      0.0211543  |
| intraday_pct_change_6              |  0.00922718 |     0.0199352  |      0.0154407  |
| intraday_rolling_std_5             | -0.00929529 |     0.0246064  |      0.0264212  |
| intraday_rolling_money_5           | -0.00175834 |    -0.0044877  |     -0.00455773 |
| intraday_rolling_money_6_3         | -0.00369373 |    -0.00865403 |     -0.009164   |
| intraday_rolling_volume_5          | -0.00169721 |    -0.00431745 |     -0.00442322 |
| intraday_rolling_volume_6_3        | -0.00365949 |    -0.00860967 |     -0.00912756 |
| intraday_pct_volatility            |  0.00960336 |     0.0396089  |      0.0383     |
| intraday_rolling_pct_volatility_5  |  0.0118225  |     0.0395367  |      0.0425203  |
| intraday_rolling_pct_volatility_10 |  0.00827556 |     0.0335309  |      0.0334182  |
| intraday_pct_vwap_low              |  0.0097716  |    -0.00987224 |     -0.00514639 |
| intraday_pct_vwap_high             | -0.00828736 |     0.015894   |      0.0109779  |
| intraday_daily_percent             |  0.0473394  |     0.025758   |      0.0216949  |

可以发现，5min/10min的相关性还是有一些的，说明技术指标短期内可以起到一些预测作用，这也给我们做**日内高频可转债**提供了一个思路。

## 特征转存数据库

这里以日内特征为例，构建数据表，应用上述特征抽取函数，提取技术面特征，然后存储到Clickhouse数据库中

```python
import pandas as pd
from tqdm import tqdm
import numpy as np
from clickhouse_driver import Client

storage_client = Client('10.0.16.11', password='******', settings={'use_numpy': True})

def make_cn_intraday_set(start_date='2018-01-01', end_date='2022-07-01'):
    '''
    主要股票：日内特征
    '''
    create_table_intraday_cn_data = '''
        create table if not exists feature.intraday_cn_data
        (
            trade_time DateTime('Asia/Shanghai'), code String, trade_day Date, minutes_of_day Int16,
            intraday_RSI_5 Float32, intraday_CMO_3 Float32, intraday_NATR_3 Float32, intraday_RVI_5 Float32, intraday_K_9_3 Float32, intraday_D_9_3 Float32, intraday_J_9_3 Float32, intraday_ADX_5 Float32, intraday_DMP_5 Float32, intraday_DMN_5 Float32, intraday_MFI_5 Float32, intraday_PVI_5 Float32, intraday_NVI_5 Float32, intraday_WILLR_5 Float32, intraday_WILLR_10 Float32, intraday_CMF_10 Float32, intraday_DPO_5 Float32, intraday_LOGRET_5 Float32, intraday_ZS_5 Float32, intraday_pct_change_3 Float32, intraday_pct_change_6 Float32, intraday_rolling_std_5 Float32, intraday_rolling_money_5 Float32, intraday_rolling_money_6_3 Float32, intraday_rolling_volume_5 Float32, intraday_rolling_volume_6_3 Float32, intraday_pct_volatility Float32, intraday_rolling_pct_volatility_5 Float32, intraday_rolling_pct_volatility_10 Float32, intraday_pct_vwap_low Float32, intraday_pct_vwap_high Float32, intraday_daily_percent Float32
        )
        ENGINE = ReplacingMergeTree()
        ORDER BY (trade_day, trade_time, code)
    '''
    storage_client.execute(create_table_intraday_cn_data)

    create_table_intraday_cn_label = '''
        create table if not exists feature.intraday_cn_label
        (
            trade_time DateTime('Asia/Shanghai'), code String, trade_day Date, minutes_of_day Int16,
            label_d0_25 Float32, label_d0_50 Float32, label_d0_75 Float32,  label_rolling_5_50 Float32,  label_rolling_10_50 Float32,  label_rolling_5_50_bps Float32,  label_rolling_10_50_bps Float32
        )
        ENGINE = ReplacingMergeTree()
        ORDER BY (trade_day, trade_time, code)
    '''
    storage_client.execute(create_table_intraday_cn_label)

    def make(code):
        # 读取K线
        kline_df = load_jqdata_kline(code=code, start_date=start_date, end_date=end_date)
        if len(kline_df) < 1:
            return
        # 生成特征和标签
        introday_data, label_data = make_tech_feature_intraday_jqdata(kline_df)
        # 分批写入，降低性能压力
        if len(introday_data) > 0:
            for i in range(int(np.ceil(len(introday_data) / 2000))):
                storage_client.insert_dataframe('INSERT INTO feature.intraday_cn_data  VALUES', introday_data.iloc[i * 2000: (i+1)*2000])
                storage_client.insert_dataframe('INSERT INTO feature.intraday_cn_label VALUES', label_data.iloc[i * 2000: (i+1)*2000])
    
    # 获取指定的股票列表
    tickers = get_jq_ticker(start_date, end_date)

    # 遍历股票，生成特征
    for ticker in tqdm(tickers): 
            make(ticker)
```

至此，我们完成了技术特征数据的提取、存储工作。

---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。

![](/img/qrcode.jpg)