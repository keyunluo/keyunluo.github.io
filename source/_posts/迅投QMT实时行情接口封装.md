---
title: 迅投QMT量化平台实时行情接口封装
date: 2022-08-02 20:00:00
tags: 
    - 迅投QMT
    - 量化交易
    - 行情接口
categories:
    - 量化
---

> 迅投QMT量化平台的xtquant库提供了python操作API，我们利用其提供的全推行情能力封装成独立的实时行情服务，实现tick、1Min等粒度的行情加工，为后续实时因子和交易信号生成提供基础保障。


## 行情监控标的

QMT支持的市场有A股股票、指数、ETF、上证50/沪深300ETF期权、国内期货、国内期货期权、香港股票等，这里先设置好待监控标的，并定期更新。

### A股股票指数以及ETF

```python
def get_a_index_etf():
    # 上证指数、深证成指、创业板指、科创50、上证50、沪深300、中证500、中证1000
    indexes = ['000001.SH', '399001.SZ', '399006.SZ', '000688.SH', '000016.SH', '000300.SH', '000905.SH', '000852.SH']
    etf = ["512100.SH", "510050.SH", "510300.SH", "513050.SH", "515790.SH", "563000.SH", "588000.SH", "513180.SH", "513060.SH", "159915.SZ", "512880.SH", "512010.SH", "512660.SH", "159949.SZ", "510500.SH", "512690.SH", "518880.SH", "511260.SH", "512480.SH", "512200.SH", "515030.SH", "511380.SH", "512000.SH", "510330.SH", "513130.SH", "513500.SH", "513100.SH", "512800.SH", "512760.SH", "159920.SZ", "159605.SZ", "159941.SZ", "162411.SZ", "513330.SH", "510900.SH", "513090.SH", "513550.SH"]

    return indexes + etf 
```

### A股重要指数成分股

```python
import akshare

def get_a_index_component():
    #指数：沪深300(000300)、中证500(000905)、中证1000(000852)
    stocks = []
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

### ETF期权

### 期货合约

### 期货期权

### 外盘期货

```python
# 外盘期货市场关注标的
def get_future_market_ticker():
    # 汇率、利率、大宗商品
    forex = ['DXY.OTC', 'EURUSD.OTC', 'GBPUSD.OTC', 'USDJPY.OTC', 'USDRUB.OTC', 'USDCNH.OTC', 'USDHKD.OTC']
    interest = ['US10YR.OTC', 'DE10YR.OTC', 'UK10YR.OTC', 'CN10YR.OTC', 'JP10YR.OTC', 'US5YR.OTC', 'US2YR.OTC', 'US1YR.OTC', 'US30YR.OTC', 'FR10YR.OTC', 'CN5YR.OTC', 'CN2YR.OTC', 'CN1YR.OTC', 'CN7YR.OTC']
    commodity = ['USHG.OTC', 'UKAH.OTC', 'UKCA.OTC', 'UKNI.OTC', 'UKPB.OTC', 'UKZS.OTC', 'UKSN.OTC', 'USZC.OTC', 'USZW.OTC', 'USYO.OTC', 'USZS.OTC', 'USLHC.OTC', 'UKOIL.OTC', 'USCL.OTC', 'USNG.OTC', 'XAUUSD.OTC', 'USGC.OTC', 'XAGUSD.OTC', 'USSI.OTC', 'AUTD.SGE', 'AGTD.SGE', 'PT9995.SGE', 'USPL.OTC', 'USPA.OTC']
    index = ["US500F.OTC", "VIXF.OTC", "US30F.OTC", "USTEC100F.OTC", "JP225F.OTC", "EU50F.OTC", "DE30F.OTC", "FR40F.OTC", "ES35F.OTC", "AU200F.OTC", "STOXX50F.OTC"]
    return forex + interest + commodity + index
```


#### 推荐阅读

- [初探迅投QMT极简策略系统](https://mp.weixin.qq.com/s/5XI09nyStjmD0faYs9UIlw)

---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。

![](/img/qrcode.jpg)


