---
title: 基于xtdata实时行情生成技术因子
date: 2022-09-11 08:00:00
tags: 
    - 迅投QMT
    - 量化交易
    - 行情接口
categories:
    - 量化
---

> 在上文实时行情服务的基础上，本文继续介绍如何将研究阶段的离线因子计算盘中实时化，以供策略使用。

## 一、行情数据缓存

### 1.1 获取A股市场所有ticker
获取沪深指数、沪深A股、沪深债券、板块指数和沪深基金中的所有股票：
```python
def get_all_a_tickers():
    # 沪深指数
    index_ticker = xtdata.get_stock_list_in_sector("沪深指数")
    # 沪深A股
    stock_ticker = xtdata.get_stock_list_in_sector("沪深A股")
    # 沪深债券
    bond_ticker = xtdata.get_stock_list_in_sector("沪深债券")
    # 板块指数
    sector_ticker = xtdata.get_stock_list_in_sector("板块指数")
    # 沪深基金
    fund_ticker = xtdata.get_stock_list_in_sector("沪深基金")
    tickers = index_ticker + stock_ticker + bond_ticker + sector_ticker + fund_ticker
    return tickers
```

### 1.2 封装下载接口
下载时可选数据的起始时间，这里按日期输入，形式为'20220909000001',意味着补充20220909日之后(包含当天)的历史1分钟K线。如果不想下载所有的股票，可替换下面代码中的`all_a_tickers`，比如沪深300(`hs300_component`)、中证500(`csi500_component`)、中证1000(`csi1000_component`)等

```python
@api.route('/download/kline/1m', methods=['GET'])
async def download_kline_1m(request):
    '''
    下载A股市场全部1分钟K线
    '''
    start_time = request.args.get("start_time", datetime.datetime.now().strftime("%Y%m%d000001"))
    
    for ticker in tqdm(all_a_tickers):
       xtdata.download_history_data(stock_code=ticker, period='1m', start_time=start_time, end_time='')
    return response.json({"data": len(all_a_tickers)})
```

### 1.3 请求测试

在浏览器中请求：`http://127.0.0.1:7800/api/xtdata/download/kline/1m?start_time=20160101000001`，便可下载2016年之后的所有数据(因请求所有数据，等待时间较长)，后面要更新的话，只需每天盘前请求一下，不需传入参数`start_time`。

## 二、实时因子计算

### 2.1 获取昨日收盘价

由于盘中请求的数据不包含昨天的收盘价，因此需要每天提前获取昨天的价格，以便方便计算当日涨跌幅：

```python
def get_last_day_price(tickers=['159919.SZ', '510050.SH', '000810.SZ'], trade_day=datetime.date.today().strftime("%Y%m%d")):
    kline_data = xtdata.get_market_data(field_list=['close'], stock_list=tickers, period='1m', start_time='', end_time=trade_day + '080000', count=1, dividend_type='front', fill_data=True)
    result = {}
    for ticker in tickers:
        result[ticker] = kline_data['close'].loc[ticker].values[0]
    return result
```

### 2.2 日内实时特征计算

我们将之前离线研究阶段的相关技术因子改用xtdata实时获取，考虑到特征数据的一致性，这里每次产生的因子都进行了放缩操作，在应用到模型中在进行逆操作，这里大家可以改成自己的实现方式。可传入多个ticker，用`,`将他们拼接到一起,如"159919.SZ,510050.SH,000810.SZ"

```python
@api.route('/feature/tech', methods=['GET'])
async def feature_tech(request, tickers=''):
    '''
    计算实时技术特征
    '''
    if tickers == '':
        tickers = request.args.get("tickers", "159919.SZ,510050.SH,000810.SZ")
    stock_list = tickers.split(',')
    start_time = request.args.get("start_time", datetime.datetime.now().strftime("%Y%m%d000001"))
    end_time = request.args.get("end_time", datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    kline_data = xtdata.get_market_data(field_list=['time', 'open', 'high', 'low', 'close', 'volume', 'amount'], stock_list=stock_list, period='1m', start_time=start_time, end_time=end_time)

    features = []
    for stock in stock_list:
        kline_df = pd.concat([kline_data[i].loc[stock].T for i in ['time', 'open', 'high', 'low', 'close', 'volume', 'amount']], axis=1)
        kline_df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'amount']
        kline_df['trade_time'] = kline_df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
        ticker= stock.split('.')[0] + {'SH': '.XSHG', 'SZ': '.XSHE'}.get(stock.split('.')[1])
        trade_date = kline_df['trade_time'].iloc[0].strftime("%Y-%m-%d")
        trade_time = int(kline_df['time'].iloc[-1] // 1000)
        kline_df['price_last'] = last_day_price.get(stock, None)
        kline_df['minute_avg'] = (kline_df['high'] + kline_df['low']) / 2
        kline_df['minutes_of_day'] = kline_df.trade_time.dt.hour * 60 + kline_df.trade_time.dt.minute

        kline_df['price_open'] = kline_df['minute_avg'].iloc[0]
        kline_df['pct_daily'] = (kline_df['close'] - kline_df['price_last']).div(kline_df['price_last'])
        kline_df['pct_intraday'] = (kline_df['close'] - kline_df['price_open']).div(kline_df['price_open'])

        pct_daily = kline_df['pct_daily'].iloc[-1]
        pct_intraday = kline_df['pct_intraday'].iloc[-1]
        rsi_3 = (kline_df.ta.rsi(length=3) / 100).fillna(0.5).iloc[-1]
        cmo_5 = (kline_df.ta.cmo(length=5) / 100).fillna(0.).iloc[-1]
        cmo_8 = (kline_df.ta.cmo(length=8) / 100).fillna(0.).iloc[-1]
        kdj_9_3 = (kline_df.ta.kdj(min(9, len(kline_df)), 3) / 100).fillna(0.5).iloc[-1].tolist() 
        willr_3 = (kline_df.ta.willr(length=3) / 100).clip(-1, 1).fillna(-0.5).iloc[-1]
        willr_5 = (kline_df.ta.willr(length=5) / 100).clip(-1, 1).fillna(-0.5).iloc[-1]
        willr_10 = (kline_df.ta.willr(length=min(10, len(kline_df))) / 100).clip(-1, 1).fillna(-0.5).iloc[-1]
        dpo_5 = (kline_df.ta.dpo(length=5, lookahead=False) * 10).clip(-3, 3).fillna(0.0).iloc[-1]
        log_return_10 = (kline_df.ta.log_return(length=10) * 10).clip(-3, 3).fillna(0.).iloc[-1]
        log_return_5 = (kline_df.ta.log_return(length=5) * 10).clip(-3, 3).fillna(0.).iloc[-1]
        log_return_3 = (kline_df.ta.log_return(length=3) * 10).clip(-3, 3).fillna(0.).iloc[-1]
        zscore_10 = (kline_df.ta.zscore(length=10)).clip(-3, 3).fillna(0.).iloc[-1]
        zscore_5 = (kline_df.ta.zscore(length=5)).clip(-3, 3).fillna(0.).iloc[-1]
        zscore_3 = (kline_df.ta.zscore(length=3)).clip(-3, 3).fillna(0.).iloc[-1]
        pct_volatility = (10 * (kline_df['high'] - kline_df['low']).div(kline_df['minute_avg'])).clip(-1, 1).fillna(0.).iloc[-1]
        rolling_pct_volatility_3 =  (20 * (kline_df['high'].rolling(3, min_periods=1).max() - kline_df['low'].rolling(3, min_periods=1).min()).div(kline_df['minute_avg'])).clip(-3, 3).fillna(0.).iloc[-1]
        rolling_pct_volatility_5 =  (20 * (kline_df['high'].rolling(5, min_periods=1).max() - kline_df['low'].rolling(5, min_periods=1).min()).div(kline_df['minute_avg'])).clip(-3, 3).fillna(0.).iloc[-1]
        rolling_pct_volatility_10 =  (20 * (kline_df['high'].rolling(10, min_periods=1).max() - kline_df['low'].rolling(10, min_periods=1).min()).div(kline_df['minute_avg'])).clip(-3.1, 3.1).fillna(0.).iloc[-1]

        feature = [ticker, trade_date, trade_time] +  (np.array([pct_daily, pct_intraday, rsi_3 , cmo_5, cmo_8] + kdj_9_3 +[willr_3, willr_5, willr_10, dpo_5, log_return_3, log_return_5, log_return_10, zscore_3, zscore_5, zscore_10, pct_volatility, rolling_pct_volatility_3, rolling_pct_volatility_5, rolling_pct_volatility_10]) * 10000).clip(-2**15, 2**15-1).round().tolist()
        features.append(feature)
    return response.json({"data": features})

```

### 2.3 请求测试

```python
data = requests.get('http://127.0.0.1:7800/api/xtdata/feature/tech?start_time=20220908000001&end_time=20220908143000&tickers=000810.SZ')
print(data.json())

{'data': [['000810.XSHE', '2022-09-08', 1662618600, 213.0, -73.0, 7815.0, 2829.0, 1124.0, 5332.0, 3403.0, 9188.0, 0.0, 0.0, -2727.0, 4200.0, 237.0, 189.0, -142.0, 8321.0, 13324.0, 15665.0, 95.0, 757.0, 757.0, 1041.0]]}
```

---

至此，我们完成了在实时行情的基础上提取实时的技术面特征，搭建实时量化交易系统。完整可运行程序请点击[阅读原文](https://github.com/ai4trade/XtQuant/blob/main/src/app_xtdata.py)获取，一些细节问题，比如程序定时运行，定时爬取数据，后面有机会进一步完善。

PS：之前陆续收到不少小伙伴的私聊，为了方便大家交流，特意建了一个微信群，大家如有需要，可在公众号下方的菜单`交流群`获得入群方式。

#### 推荐阅读

- [基于xtdata搭建实时行情请求服务](https://mp.weixin.qq.com/s/1VZJXPG-o0LHTsnTeuQ49Q)
- [利用pandas_ta自动提取技术面特征](https://mp.weixin.qq.com/s/PPduk4xPcix9USW9HmUpHw)

---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。

![](/img/qrcode.jpg)

