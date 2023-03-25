---
title: 华鑫奇点API L2行情使用初体验
date: 2023-03-25 18:00:00
tags: 
    - 量化交易
categories:
    - 华鑫证券
    - 奇点API
    - 量化交易
    - 行情接口
---

> 华鑫证券提供了供个人低成本使用的量化交易API，本文简要介绍其基本情况以及Level2沪深行情功能。

# 一、基本情况

## 1.1 奇点API

奇点柜台是华鑫证券自研的证券交易柜台，与期货市场的CTP类似，提供了开放式API接口，包括C/C++、Java、C#、Python等主要语言，支持股票、两融、期权、期货的行情与交易。官方提供了仿真交易环境以及7×24小时的测试环境，供认证和测试使用。

官方网址是：https://n-sight.com.cn/

![](/img/nsight.png)


## 1.2 申请步骤

- nsight网站注册成为专业投资者(50W资产截图)
- 获取模拟账户，可进行模拟测试
- 实盘步骤：
  - 具有软件销售资质的营业执照，签订软件反采购合同
  - 仿真测试，获取软件标识
    - 股票：股票买卖，撤单，报撤单比例控制
    - 期权：买开、卖平、买平、卖开、撤单等
  - 交易系统评估，撰写文档
    - 股票：软件说明书、软件运维手册、交易信息系统合规性自检表
    - 期权：软件说明书、软件运维手册、软件功能承诺函(①成交持仓比例控制；②一键撤单；③程序化流速控制)、交易信息系统合规性自检表
  - 测试机器申请
    - 上海金桥机房
    - 深圳东莞机房
  - 系统测试
    - 提供的机器可访问自己提供的固定IP/端口，从而与外界通信
    - 共享服务器为CentOS7物理机，多个用户抢占式使用所有资源

实盘机房每个都分A/B两类：
- A类量化：券商提供机器，免费提供Level 2 行情，独占服务器有采购费用和创收要求，共享服务器免费
- B类量化：自行准备机器，需采购Level 2 行情，独占服务器有创收要求，共享服务器免费

需要注意的是，使用A类共享服务器奇点API的股票佣金比一般QMT/PTrade量化稍微贵一些，其他佣金都不变。

# 二、模拟环境

## 2.1 通信模式

基于Tora Python SDK中的API和回调SPI实现程序与交易/行情系统的交互

![](/img/tora.png)

python api支持版本：
- Windows: 3.7
- Linux: 3.7以上的版本均可，包括最新的Python3.11

## 2.2 主要功能封装

行情主要是订阅功能的封装，比如基于官方的demo，我们可以调用股票和期权的L1行情，在spi回调函数中进行数据粗加工或直接落盘：

```python
class Quoter:
    def __init__(self, Counter) -> None:
        self.api = None
        self.spi = None
        self.Counter = Counter
    def login(self):
        self.api = xmdapi.CTORATstpXMdApi_CreateTstpXMdApi(xmdapi.TORA_TSTP_MST_TCP, xmdapi.TORA_TSTP_MST_TCP)
        self.spi = MdSpi(self.api)
        self.api.RegisterSpi(self.spi)
        if self.Counter == '1':
            frontAddress = "tcp://210.14.72.21:4402" 
        elif self.Counter == '2':
            frontAddress = "tcp://210.14.72.16:9402" 
        self.api.RegisterFront(frontAddress)
        print('frontAddress[%s]' % frontAddress)
        self.api.Init()
    def Join(self):
        if self.api is not None:
            self.api.Join()
    def subscribe_stock(self, code_list=[]):
        self.spi.subscribe_stock(code_list)
    def unsubscribe_stock(self, code_list=[]):
        self.spi.unsubscribe_stock(code_list)
    def subscribe_option(self, code_list=[]):
        self.spi.subscribe_option(code_list)
    def unsubscribe_option(self, code_list=[]):
        self.spi.unsubscribe_option(code_list)
    def subscribe_rapid(self, code_list=[]):
        self.spi.subscribe_rapid(code_list)
    def unsubscribe_rapid(self, code_list=[]):
        self.spi.unsubscribe_rapid(code_list)
    def GetApiVersion():
        return xmdapi.CTORATstpXMdApi_GetApiVersion()
```

例子：打印实时期权5档行情回调功能：
```python
    def OnRtnSPMarketData(self, pMarketDataField):
        print("TradingDay[%s] UpdateTime[%s] UpdateMillisec[%s] SecurityID[%s] SecurityName[%s] LastPrice[%.3f] Volume[%d] Turnover[%.3f] BidPrice1[%.3f] BidVolume1[%d] AskPrice1[%.3f] AskVolume1[%d]  BidPrice2[%.3f] BidVolume2[%d] AskPrice2[%.3f] AskVolume2[%d] UpperLimitPrice[%.3f] LowerLimitPrice[%.3f]"
            % (pMarketDataField.TradingDay, pMarketDataField.UpdateTime, pMarketDataField.UpdateMillisec, pMarketDataField.SecurityID, pMarketDataField.SecurityName, pMarketDataField.LastPrice, pMarketDataField.Volume,
               pMarketDataField.Turnover, pMarketDataField.BidPrice1, pMarketDataField.BidVolume1, pMarketDataField.AskPrice1,
               pMarketDataField.AskVolume1, pMarketDataField.BidPrice2, pMarketDataField.BidVolume2, pMarketDataField.AskPrice2,
               pMarketDataField.AskVolume2, pMarketDataField.UpperLimitPrice, pMarketDataField.LowerLimitPrice))
```

结果：实时打印行情
![](/img/option_spi.png)

## 2.3 测试环境

- 仿真测试环境：同交易所交易时间，价格同步交易所，一般延迟3秒，开盘延迟可能略长一点。成交数量则包含其他仿真参与者撮合成交量。

- 7*24测试环境：供测试使用，行情不完整(无指数行情)，历史某日数据的播放。

# 三、行情测试

## 3.1 延迟测试

- 交易服务器：ping值在20微秒左右
- 行情服务器：TCP延迟在90微秒左右，另外提供了低延时的UDP组播行情

## 3.2 L2行情

L2沪&深行情使用UDP组播的方式获取：

``` python
    api = lev2mdapi.CTORATstpLev2MdApi_CreateTstpLev2MdApi(lev2mdapi.TORA_TSTP_MST_MCAST)
    api.RegisterMulticast(LEV2MD_MCAST_FrontAddress, LEV2MD_MCAST_InterfaceIP, "")
    print("LEV2MD_MCAST_FrontAddress[UDP]::%s" % LEV2MD_MCAST_FrontAddress)
    print("LEV2MD_MCAST_InterfaceIP::%s" % LEV2MD_MCAST_InterfaceIP)
```

可订阅单个标的，也可订阅全市场的L2行情。

### 3.2.1 L2 Tick 10档行情

![](/img/l2_tick.png)

### 3.2.2 L2 逐笔委托/成交行情

![](/img/l2_weituo.png)

## 结论

整体来看华鑫奇点API提供了免费机房托管途径，是对个人用户最友好的一种量化接入方式。内网环境降低了延迟，一个API可以交易股票、转债、两融、期权、期货等多个品种，部分机房的免费L2行情更是提供了高频交易的机会。

---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。
![](/img/qrcode.jpg)

