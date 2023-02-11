---
title: QMT量化平台常见问题QA2
date: 2023-02-11 20:00:00
tags: 
    - 资源
    - 量化交易
    - 量化投研
categories:
    - 量化
---

> 续2022年11月发布的《QMT量化平台常见问题QA》，更新近期典型问题。

### 2.21 为什么MiniQMT无法获取/下载行情了

- 迅投的行情问题，默认的迅投上海站点`211.152.57.213`有问题，需要手动更改为下面那个:`211.152.57.214`
- 更改路径：主QMT-下方行情图标-手动选择

### 2.22 MiniQMT支持新版本的Python吗

- 最新已经支持python3.6-3.11了
- 更新方式：主QMT-设置-交易设置-模型设置里更新Python库
- 主QMT仍然为3.6版本，无法使用新版Python

### 2.23 如何在主QMT中安装软件包

- 可以在Python官网下载一个免安装软件包: `python-3.6.8-embed-amd64.zip`，解压后将python可执行文件放到QMT的`bin.x64`目录中
- 然后打开`cmd`，切换到`bin.x64`目录下，将`https://bootstrap.pypa.io/get-pip.py` 文件一并下载，然后`.\python get-pip.py` 安装pip
- 安装第三方包： `.\python -m pip install ipython`

### 2.24 MiniQMT和主QMT可以共享数据吗

- 可以在xtquant的软件包中设置默认的数据路径，具体来说，在`Lib\site-packages\xtquant\xtdata.py`中修改自定义的数据路径`init_data_dir`
- 可以设为一个外部的路径，方便多个QMT间共享数据

### 2.25 xtdata的输出不符合习惯，可以改吗

- 同上，可以直接修改`xtdata.py`中的函数
- 例如，直接获取`get_market_data_ori`的`client.get_market_data3`结果，然后不做pandas转换，直接写到数据库中

### 2.26 QMT支持哪些品种交易

- 股票、两融、期权、期货，其中期权需要做程序化交易备案，部分券商也支持期货的交易

### 2.27 QMT可以自动每天登录吗

- 部分券商版本重启后不会保存密码或弹出验证码，需要用外挂脚本手动定时启动，有群友已经共享出来了

### 2.28 QMT支持VNPY吗

- 有群友已经制作了pypi包：https://github.com/fsksf/vnpy_qmt

### 2.29 新版本sanic框架运行出错

- sanic22版本更新较多，可以把`app.run`之前的app代码移动到`__main__`的外面
- workers设为1，禁止多进程
- 中文无法正常显示：`return  response.json(xxx, ensure_ascii=False)`

### 2.30 pandas单线程处理太慢了，如何加速

- pandarallel：利用multiprocessing开启多进程
- 异步框架如sanic: 利用异步+多进程的方式加速
- 数据库：dophindb、Clickhouse等存储系统的向量计算

### 2.31 QMT回测准吗

- 回测最细的粒度为1分钟，假设该分钟开始就成交了，因此实际操作会存在一定的滑点问题，不能完全依赖回测，可以实盘前先模拟运行

### 2.32 价格笼子对量化有什么影响

- 不能无脑发涨跌停单了，超过价格笼子直接变成废单
- 利用5挡或L2行情判断价格上下限

### 2.33 QMT可以使用通达信公式吗

- 不可以直接用，需要用`https://github.com/mpquant/MyTT` 库进行转换

### 2.34 QMT能获取低于3s的行情吗

- QMT提供的是L1行情，最短时间间隔是3s，若成交不活跃，某一分钟内可能都没有行情数据
- L2行情提供了逐笔委托、逐笔成交行情，获取的数据频次更高
- QMT实盘运行时每3s都会刷新一次，因此即时做1分钟的K线，也要记录该分钟是否已经运行过了，避免重复下单
- 国内期货的L1行情刷新频次为0.5s


#### 推荐阅读

- [QMT量化平台常见问题QA](https://mp.weixin.qq.com/s/IzGlj6tnKQnKYH9nhIuWsg)

PS：之前陆续收到不少小伙伴的私聊，为了方便大家交流，特意建了一个微信群，大家如有需要，可在公众号下方的菜单`交流群`获得入群方式。

---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。

![](/img/qrcode.jpg)