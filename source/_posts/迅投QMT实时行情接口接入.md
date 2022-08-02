---
title: 迅投QMT实时行情接口接入
date: 2022-08-02 20:00:00
tags: 
    - 迅投QMT
    - 量化交易
    - 行情接口
categories:
    - 量化
---

> 迅投QMT量化平台的xtquant库提供了python操作API，我们利用其提供的全推行情能力封装成独立的实时行情服务，实现tick、1Min等粒度的行情加工，为后续实时因子和交易信号生成提供基础保障。


## 一、标的监控

QMT支持的市场有A股股票、指数、ETF、上证50/沪深300ETF期权、国内期货、国内期货期权、香港股票等，这里先设置好待监控标的，并定期更新。

### 1.1 A股股票指数以及ETF

获取 上证指数、深证成指、创业板指、科创50、上证50、沪深300、中证500、中证1000等主要指数ticker，以及活跃ETFticker

```python
def get_a_index_etf():
    indexes = ['000001.SH', '399001.SZ', '399006.SZ', '000688.SH', '000016.SH', '000300.SH', '000905.SH', '000852.SH']
    etf = ["512100.SH", "510050.SH", "510300.SH", "513050.SH", "515790.SH", "563000.SH", "588000.SH", "513180.SH", "513060.SH", "159915.SZ", "512880.SH", "512010.SH", "512660.SH", "159949.SZ", "510500.SH", "512690.SH", "518880.SH", "511260.SH", "512480.SH", "512200.SH", "515030.SH", "511380.SH", "512000.SH", "510330.SH", "513130.SH", "513500.SH", "513100.SH", "512800.SH", "512760.SH", "159920.SZ", "159605.SZ", "159941.SZ", "162411.SZ", "513330.SH", "510900.SH", "513090.SH", "513550.SH"]
    return indexes + etf 
```

### 1.2 A股重要指数成分股

获取沪深300(000300)、中证500(000905)、中证1000(000852)指数成分股

```python
import akshare

def get_a_index_component():
    hs300 = akshare.index_stock_cons_weight_csindex(symbol="000300")
    hs300['stock'] = hs300.apply(lambda row: row['成分券代码'] + '.' + {'上海证券交易所' :'SH', '深圳证券交易所': 'SZ'}.get(row['交易所']), axis=1)
    csi500 = akshare.index_stock_cons_weight_csindex(symbol="000905")
    csi500['stock'] = csi500.apply(lambda row: row['成分券代码'] + '.' + {'上海证券交易所' :'SH', '深圳证券交易所': 'SZ'}.get(row['交易所']), axis=1)
    csi1000 = akshare.index_stock_cons_weight_csindex(symbol="000852")
    csi1000['stock'] = csi1000.apply(lambda row: row['成分券代码'] + '.' + {'上海证券交易所' :'SH', '深圳证券交易所': 'SZ'}.get(row['交易所']), axis=1)

    hs300_component = hs300.set_index('stock')[['指数代码', '成分券名称', '权重']].to_dict('index')
    csi500_component = csi500.set_index('stock')[['指数代码', '成分券名称', '权重']].to_dict('index')
    csi1000_component = csi1000.set_index('stock')[['指数代码', '成分券名称', '权重']].to_dict('index')

    return {**hs300_component, **csi500_component, **csi1000_component}
```

### 1.3 ETF期权

获取50ETF、300ETF对应的当月/次月期权ticker

```python
import re, math, datetime

def get_etf_option():
    select_option = {}

    etf_price = {}
    # 获取ETF行情
    etf_tick = xtdata.get_full_tick(['510300.SH', '510050.SH', '159919.SZ'])
    # 取今日开盘价/昨日收盘价均值
    for code in ['510300.SH', '510050.SH', '159919.SZ']:
        etf_price[code] = (etf_tick[code]['open'] + etf_tick[code]['lastClose']) / 2

    options = xtdata.get_stock_list_in_sector('上证期权') + xtdata.get_stock_list_in_sector('深证期权')
    # 获取主力期权(标的价格附近上下5档,当月/次月)
    option_data = []
    for code in options:
        meta = xtdata.get_instrument_detail(code)
        # 期权名称
        name = meta['InstrumentName']
        # 对应的ETF
        etf = re.findall(r'\((\d+)\)', meta['ProductID'])[0] 
        etf = {'510300': '510300.SH', '510050': '510050.SH', '159919': '159919.SZ'}.get(etf)
        # 剩余有效日
        days = (datetime.date(year=int(str(meta['ExpireDate'])[:4]), month=int(str(meta['ExpireDate'])[4:6]), day=int(str(meta['ExpireDate'])[6:8])) - datetime.date.today()).days
        call_put = 'call' if '购' in name else 'put'
        if days < 32:
            if math.fabs(etf_price[etf] - int(name[-4:]) / 1000.0) < 0.2:
                select_option[code] = [etf, call_put, int(name[-4:]), days]
        elif days < 65:
            if math.fabs(etf_price[etf] - int(name[-4:]) / 1000.0) < 0.25:
                select_option[code] = [etf, call_put, int(name[-4:]), days]

    return select_option
```

### 1.4 期货合约

获取中金所、大商所、郑商所、上期所的主要连续合约

```python
import re
def get_main_contract():
    contract = xtdata.get_stock_list_in_sector('中金所') + xtdata.get_stock_list_in_sector('大商所') + xtdata.get_stock_list_in_sector('郑商所') + xtdata.get_stock_list_in_sector('上期所') 
    market_mapping = {'CZCE': 'ZF', 'DCE': 'DF', 'CFFEX': 'IF', 'SHFE': 'SF'}
    contract_main = [i.split('.')[0] + '.' + market_mapping.get(i.split('.')[1]) for i in contract if re.search(r'[A-Za-z]00\.[A-Z]', i)]
    if 'IM00.IF' not in contract_main:
        contract_main.append('IM00.IF')
    return contract_main
```

### 1.5 外盘期货

外盘期货市场主要标的，包含汇率、利率、商品以及股指

```python
def get_future_market_ticker():
    forex = ['DXY.OTC', 'EURUSD.OTC', 'GBPUSD.OTC', 'USDJPY.OTC', 'USDRUB.OTC', 'USDCNH.OTC', 'USDHKD.OTC']
    interest = ['US10YR.OTC', 'DE10YR.OTC', 'UK10YR.OTC', 'CN10YR.OTC', 'JP10YR.OTC', 'US5YR.OTC', 'US2YR.OTC', 'US1YR.OTC', 'US30YR.OTC', 'FR10YR.OTC', 'CN5YR.OTC', 'CN2YR.OTC', 'CN1YR.OTC', 'CN7YR.OTC']
    commodity = ['USHG.OTC', 'UKAH.OTC', 'UKCA.OTC', 'UKNI.OTC', 'UKPB.OTC', 'UKZS.OTC', 'UKSN.OTC', 'USZC.OTC', 'USZW.OTC', 'USYO.OTC', 'USZS.OTC', 'USLHC.OTC', 'UKOIL.OTC', 'USCL.OTC', 'USNG.OTC', 'XAUUSD.OTC', 'USGC.OTC', 'XAGUSD.OTC', 'USSI.OTC', 'AUTD.SGE', 'AGTD.SGE', 'PT9995.SGE', 'USPL.OTC', 'USPA.OTC']
    index = ["US500F.OTC", "VIXF.OTC", "US30F.OTC", "USTEC100F.OTC", "JP225F.OTC", "EU50F.OTC", "DE30F.OTC", "FR40F.OTC", "ES35F.OTC", "AU200F.OTC", "STOXX50F.OTC"]
    return forex + interest + commodity + index
```

## 二、行情获取

QMT的实时行情获取主要分为tick分笔数据和kline分钟数据，其中全推tick行情仅支持股票，其他tick/kline行情则需要通过订阅的方式获得。

### 2.1 全推方式获得tick行情

首先订阅全推行情：
```python
a_index_etf = get_a_index_etf()
subscribe_id = xtdata.subscribe_whole_quote(a_index_etf)
```

然后实时查询,整理成dataframe：
```python
data = xtdata.get_full_tick(a_index_etf)
df = pd.DataFrame(data)
```

抽出几列数据，可看出指数无五档行情，一般股票则包含了实时请求的快照：

|                     | 000300.SH         | 510300.SH                           | 159915.SZ             |
|:--------------------|:------------------|:------------------------------------|:----------------------|
| timetag             | 20220802 14:57:07 | 20220802 14:59:33                   | 20220802 14:59:33     |
| lastPrice           | 4106.204          | 4.167                               | 2.605                 |
| open                | 4144.377          | 4.21                                | 2.627                 |
| high                | 4144.377          | 4.21                                | 2.637                 |
| low                 | 4071.431          | 4.132                               | 2.575                 |
| lastClose           | 4188.678          | 4.238                               | 2.657                 |
| amount              | 278109651400      | 2262356300                          | 937252200             |
| volume              | 140760397         | 5434039                             | 3594548               |
| pvolume             | 0                 | 0                                   | 0                     |
| stockStatus         | 0                 | 0                                   | 0                     |
| openInt             | 0                 | 13                                  | 18                    |
| settlementPrice     | 0                 | 0                                   | 0                     |
| lastSettlementPrice | 0                 | 0                                   | 0                     |
| askPrice            | [0, 0, 0, 0, 0]   | [4.168, 4.169, 4.17, 4.171, 4.172]  | [2.605, 0, 0, 0, 0]   |
| bidPrice            | [0, 0, 0, 0, 0]   | [4.167, 4.166, 4.165, 4.164, 4.163] | [2.605, 0, 0, 0, 0]   |
| askVol              | [0, 0, 0, 0, 0]   | [2715, 452, 824, 577, 322]          | [20672, 733, 0, 0, 0] |
| bidVol              | [0, 0, 0, 0, 0]   | [1904, 11190, 4558, 2520, 3142]     | [20672, 0, 0, 0, 0]   |


### 2.2 订阅方式获得tick/kline行情

首先订阅指定ticker，如有多个则依次订阅, count参数指定缓存数据量大小:
```python
contract = get_main_contract()
subscribe_ids = []
for ticker in contract:
    subscribe_ids.append(xtdata.subscribe_quote(ticker, 'tick', count=10))
    subscribe_ids.append(xtdata.subscribe_quote(ticker, '1m', count=10))
```

查询tick行情：
```python
tick_data = xtdata.get_market_data(stock_list=['IC00.IF', 'IM00.IF', 'ag00.SF'], period='tick', count=5)
print(pd.DataFrame(tick_data['IC00.IF']))
```
可以看到，tick行情时间戳精确到毫秒，字段同全推行情。

查询Kline行情：
```python
kline_data =  xtdata.get_market_data(stock_list=['90001156.SZO'], period='1m', count=10)
df = pd.concat([kline_data[i].T for i in ['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'settelementPrice', 'openInterest']], axis=1)
df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'settelementPrice', 'openInterest']
df = df[df.volume !=0]
df['time'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
```

| time                |   open |   high |    low |   close |   volume |   amount |   settelementPrice |   openInterest |
|:--------------------|-------:|-------:|-------:|--------:|---------:|---------:|-------------------:|---------------:|
| 2022-08-02 14:56:00 | 0.1728 | 0.1739 | 0.1728 |  0.1739 |       40 |    69333 |             0      |          16019 |
| 2022-08-02 14:57:00 | 0.1739 | 0.1741 | 0.1729 |  0.1741 |       77 |   133675 |             0      |          15959 |
| 2022-08-02 15:00:00 | 0.1741 | 0.1741 | 0.1731 |  0.1731 |        1 |     1731 |             0.1731 |          15960 |


​至此我们完成了行情获取的基本操作，下文将继续介绍如何在实时行情中提取技术因子，设计定时任务，构造机器学习工作流。

#### 推荐阅读

- [初探迅投QMT极简策略系统](https://mp.weixin.qq.com/s/5XI09nyStjmD0faYs9UIlw)

---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。

![](/img/qrcode.jpg)


