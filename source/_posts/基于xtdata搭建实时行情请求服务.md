---
title: 基于xtdata搭建实时行情请求服务
date: 2022-09-10 08:00:00
tags: 
    - 迅投QMT
    - 量化交易
    - 行情接口
categories:
    - 量化
---

> 本文介绍如何将xtquant的mini客户端的行情功能以Web服务的形式暴露出来供查询，并给出一个完整可运行的案例。

## 一、准备工作

- 开启客户端：XtMiniQmt.exe，可无需登录，可在云服务器中开启
- 安装python库
  - xtquant: 拷贝至使用的Python库中
  - sanic, aiohttp: web服务框架
  - akshare: 数据源封装接口
  - pandas-market-calendars: 交易所日历
  - pandas: 数据分析工具

## 二、程序启动

在sanic中，我们可以建立一个监听器，在程序启动前执行一些全局化的操作, 这里我们可以将订阅全推市场行情放到这里，这样的话每次启动都会自动执行，订阅时返回一个订阅ID，为了避免内存泄漏，在程序结束时我们要记得使用unsubscribe_quote接口将这些订阅取消。

```python
@api.listener('before_server_start')
async def before_server_start(app, loop):
    '''全局共享session'''
    global session, subscribe_ids, hs300_component, csi500_component, csi1000_component
    jar = aiohttp.CookieJar(unsafe=True)
    session = aiohttp.ClientSession(cookie_jar=jar, connector=aiohttp.TCPConnector(ssl=False))
    subscribe_ids = []
    subscribe_ids.append(xtdata.subscribe_whole_quote(['SH', 'SZ', 'SHO', 'SZO', 'HK', 'IF', 'ZF', 'DF', 'SF']))
    hs300_component, csi500_component, csi1000_component = get_a_index_component()

@api.listener('after_server_stop')
async def after_server_stop(app, loop):
    '''关闭session'''
    for seq_num in subscribe_ids:
        xtdata.unsubscribe_quote(seq_num)
    await session.close()
```

## 三、关键功能

### 3.1 行情订阅

对单个标的，我们使用订阅的方式，可获取tick/kline行情：
```python
@api.route('/subscribe', methods=['GET'])
async def subscribe(request, ticker_input=''):
    '''
    订阅单股行情: 获得tick/kline行情
    '''
    if ticker_input == '':
        ticker = request.args.get("ticker", "000001.SH")
    else:
        ticker = ticker_input
    period = request.args.get("period", "1m")
    start_time = request.args.get("start_time", "")
    end_time = request.args.get("end_time", "")
    subscribe_ids.append(xtdata.subscribe_quote(ticker, period, start_time=start_time, end_time=end_time, count=10))
    if ticker_input == '':
        return response.json({"data": subscribe_ids[-1]})
    else:
        return {"data": subscribe_ids[-1]}
```

进一步的，我们可以传入预定义的股票池，如沪深300成分股，批量订阅：
```python
@api.route('/subscribe/kline/hs300', methods=['GET'])
async def quote_kline_hs300(request):
    '''
    订阅市场行情: 沪深300成分股1分钟K线行情
    '''
    seq_ids = []
    for ticker in hs300_component:
       seq_id =  await subscribe(request, ticker_input=ticker)
       seq_ids.append(seq_id.get('data', -1))
    return response.json({"data": seq_ids})
```

### 3.2 K线行情查询

利用`get_market_data`接口，我们读取价格和成交量等数据，然后对每个股票组装成一个DataFrame结构：
```python
@api.route('/quote/kline', methods=['GET'])
async def quote_kline(request, tickers=''):
    '''
    查询市场行情: 获得kline数据
    '''
    if tickers == '':
        tickers = request.args.get("tickers", "IM00.IF,159919.SZ,00700.HK,10004407.SHO")
    period = request.args.get("period", "1m")
    start_time = request.args.get("start_time", "")
    end_time = request.args.get("end_time", "")
    count = request.args.get("count", "1")
    dividend_type = request.args.get("dividend_type", "none") # none 不复权 front 前复权 back 后复权 front_ratio 等比前复权 back_ratio 等比后复权
    stock_list = tickers.split(',')

    kline_data = xtdata.get_market_data(field_list=['time', 'open', 'high', 'low', 'close', 'volume', 'amount'], stock_list=stock_list, period=period, start_time=start_time, end_time=end_time, count=int(count), dividend_type=dividend_type, fill_data=True)

    quote_data = {}
    for stock in stock_list:
        df = pd.concat([kline_data[i].loc[stock].T for i in ['time', 'open', 'high', 'low', 'close', 'volume', 'amount']], axis=1)
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df = df[df.volume !=0]
        df['time'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime("%Y-%m-%d %H:%M:%S"))
        df['ticker'] = stock
        df = df[['ticker', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount']].values.tolist() 
        quote_data[stock] = df

    return response.json({"data": quote_data})
```

基于上面的实现，我们可以直接将沪深300股票的查询做进一步封装：
```python
@api.route('/quote/kline/hs300', methods=['GET'])
async def quote_kline_hs300(request):
    '''
    查询市场行情: 沪深300成分股1分钟K线行情
    '''
    return await quote_kline(request, ','.join(list(hs300_component)))
```


### 3.3 Tick行情查询

由于订阅了全推行情，因此获取tick数据可直接使用`get_full_tick`接口：
```python
@api.route('/quote/tick', methods=['GET'])
async def quote_tick(request):
    '''
    查询市场行情: 获得tick数据
    '''
    tickers = request.args.get("tickers", "159919.SZ,00700.HK")
    stock_list = tickers.split(',')
    data = xtdata.get_full_tick(stock_list)
    return response.json({"data": data})

```

## 四、功能测试

启动程序，即运行`python app_xtdata.py`后，我们便可以使用`requests`库或浏览器访问数据了,
base_url = 'http://172.16.0.4:7800/api/xtdata'

- 订阅沪深300： base_url + /subscribe/kline/hs300
- 查询沪深300成份股实时行情：base_url + /quote/kline/hs300
```python
  data = requests.get(base_url + '/quote/kline/hs300').json()
  print(data['data']['600941.SH'])
  [['600941.SH', '2022-09-09 15:00:00', 65.31, 65.37, 65.31, 65.37, 192, 1255104.0]]
```
- 查询tick快照行情：base_url + /quote/tick?tickers=159919.SZ,510050.SH
- 查询合约基础信息：base_url + /quote/instrument/detail?ticker=510300.SH
```python
  {'data': {'ExchangeID': 'SH',
  'InstrumentID': '510300',
  'InstrumentName': '沪深300ETF',
  'ProductID': None,
  'ProductName': None,
  'CreateDate': '0',
  'OpenDate': '20120528',
  'ExpireDate': 99999999,
  'PreClose': 4.101,
  'SettlementPrice': 4.101,
  'UpStopPrice': 4.511,
  'DownStopPrice': 3.691,
  'FloatVolume': 10836487700.0,
  'TotalVolume': 10836487700.0,
  'LongMarginRatio': None,
  'ShortMarginRatio': None,
  'PriceTick': 0.001,
  'VolumeMultiple': 1,
  'MainContract': None,
  'LastVolume': None,
  'InstrumentStatus': 0,
  'IsTrading': None,
  'IsRecent': None}}
```

至此，我们完成了行情服务的Web封装，下篇将继续介绍如何在实时行情的基础上提取实时的技术面特征，搭建实时量化交易系统。完整可运行程序请点击阅读原文获取。

#### 推荐阅读

- [迅投QMT实时行情接口接入](https://mp.weixin.qq.com/s/cWYXulT-daBgrDtr36CHAA)

---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。

![](/img/qrcode.jpg)

