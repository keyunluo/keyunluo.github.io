---
title: 跨市场联动:基于美股隔日行情预测A股行业涨跌
date: 2022-07-20 12:00:00
tags: 
    - 跨市场联动
    - 美股
    - pandas_ta
    - 技术分析
categories:
    - 量化
---

> 随着A股北上资金的不断涌入，跨市场联动性也愈发显著，在内循环的同时也时刻受着外部重要市场行情波动的影响，美股作为全球市场，一丝风吹草动都对全球金融造成剧烈波动。本文将探索美股行情中的技术面因子对当天A股市场的行业影响，使用机器学习技术预测行业涨跌情况，并同基准沪深300指数作对比以说明实际效果。 

## 美股因子数据生成

### 美股标的选择

美股市场总有有高达数万个股票、ETF标的，我们这里选择那些具有代表性的股票和ETF作为参考标的。

- [道琼斯工业平均指数](https://www.slickcharts.com/dowjones)成分股：是在美国证券交易所上市的30家著名公司的价格加权衡量股票市场指数。
- [标普500指数](https://www.slickcharts.com/sp500)成分股：美国上市公司总市值Top500，其成分股由400种工业股票、20种运输业股票、40种公用事业股票和40种金融业股票组成。
- [纳斯达克100指数](https://www.slickcharts.com/nasdaq100)成分股：是美国纳斯达克100支最大型本地及国际非金融类上市公司组成的股市指数，以市值作基础，并以一些规则平衡较大市值股份造成的影响。
- 大盘和行业指数ETF：
  - 大宗商品：白银(SLV)、黄金(GLD)、金矿(GDX)、天然气(UNG)、太阳能(TAN)、能源ETF(XLE)、商品指数(DBC)、油矿开采(XOP)、原油基金(USO)、油气服务(OIH)
  - 大盘指数: 标普500(SPY)、道琼斯指数(DIA)、纳斯达克100(QQQ)、罗素2000(IWM)、恐慌指数(UVXY)、恐慌指数(VIXY)、价值股(VTV)、罗素1000成长(IWF) 中国大盘股(FXI)、中国海外互联网(KWEB)、日本ETF(EWJ)、台湾(EWT)、韩国(EWY)、澳大利亚(EWA)、香港(EWH)、沪深300(ASHR)、欧洲(VGK)、英国(EWU)、德国(EWG)、欧盟(EZU)、巴西(EWZ)
  - 债券：20年国债(TLT)、全债(AGG)、市政债(MUB)、通胀债(TIP)、债券指数(HYG)、短期国债(SHV)、公司债(LQD)、高价值债(JNK)、短期公司债(VCSH)、中期公司债(VCIT)、1-3年国债(SHY)、新兴市场美元债(EMB)
  - 行业：金融(XLF)、生物(XBI)、半导体(SMH)、非必须消费品(XLY)、高科技(XLK)、医疗保健(XLV)、日常消费(XLP)、公共事业(XLU)、工业指数(XLI)、房地产(IYR)、不动产(VNQ)、原料(XLB)、区域银行(KRE)、信息技术(VGT)、航空业ETF(JETS)、农产品(DBA)、零售(XRT)、金属矿业(XME)、房屋建筑(XHB)
  - 汇率: 美元(UUP) 
- 中概股代表股：阿里巴巴(BABA)、台积电(TSM)、京东(JD)、拼多多(PDD)、网易(NTES)、百度(BIDU)、理想汽车(LI)、蔚来(NIO)、小鹏汽车(XPEV)、百胜中国(YUMC)、百济神州(BGNE)、贝壳(BEKE)、携程(TCOM)、陆金所(LU)、哔哩哗啦(BILI)、腾讯音乐(TME)、富途(FUTU)、万国数据(GDS)、微博(WB)、新东方(EDU)、爱奇艺(IQ)

代码：爬取指数成分股

```python
import re, requests

def get_us_market_ticker():
    headers = {
        'authority': 'www.slickcharts.com',
        'cache-control': 'max-age=0',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'referer': 'https://www.google.com/',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7'
    }

    sp500= requests.get('https://www.slickcharts.com/sp500', headers=headers)
    nasdaq100 = requests.get('https://www.slickcharts.com/nasdaq100', headers=headers)
    dowjones30 = requests.get('https://www.slickcharts.com/dowjones', headers=headers)
    
    component_ticker = set(re.findall(r"/symbol/([A-Za-z\.]+)", sp500.text)) | set(re.findall(r"/symbol/([A-Za-z\.]+)", nasdaq100.text)) | set(re.findall(r"/symbol/([A-Za-z\.]+)", dowjones30.text))
    
    etf_ticker = set(['SLV', 'GLD', 'GDX', 'UNG', 'TAN', 'XLE', 'DBC', 'XOP', 'USO', 'OIH', 'SPY', 'DIA', 'QQQ', 'IWM', 'UVXY', 'VIXY', 'VTV', 'IWF', 'FXI', 'KWEB', 'EWJ', 'EWT', 'EWY', 'EWA', 'EWH', 'ASHR', 'VGK', 'EWU', 'EWG', 'EZU', 'EWZ', 'TLT', 'AGG', 'MUB', 'TIP', 'HYG', 'SHV', 'LQD', 'JNK', 'VCSH', 'VCIT', 'SHY', 'EMB', 'XLF', 'XBI', 'SMH', 'XLY', 'XLK', 'XLV', 'XLP', 'XLU', 'XLI', 'IYR', 'VNQ', 'XLB', 'KRE', 'VGT', 'JETS', 'DBA', 'XRT', 'XME', 'XHB', 'UUP'])
    
    cn_ticker = set(['BABA', 'TSM', 'JD', 'PDD', 'NTES', 'BIDU', 'LI', 'NIO', 'XPEV', 'YUMC', 'BGNE', 'BEKE', 'TCOM', 'LU', 'BILI', 'TME', 'FUTU', 'GDS', 'WB', 'EDU', 'IQ'])
    
    ticker = component_ticker | etf_ticker | cn_ticker
    
    return list(ticker)
```

可以获得约600个待跟踪标的。

### 技术因子生成

生成日线级别的技术因子，基于上篇介绍的pandas_ta，这里直接给出相关代码，需要注意的是要对因子进行`标准化`，限制取值范围为`[-10, 10]`，避免极端值对后续模型的影响：

```python
# 技术指标: 日间
def make_tech_feature_daily(dayline_df, index_col='trade_day'):
    # 特征
    if dayline_df.index.name != index_col:
        dayline_df = dayline_df.set_index(index_col)
    df_len = len(dayline_df)
    if df_len < 2:
        feature_df = pd.DataFrame(index=dayline_df.index, columns = ['ADX_2', 'DMP_2', 'DMN_2', 'ADX_5', 'DMP_5', 'DMN_5', 'ADX_22',
       'DMP_22', 'DMN_22', 'CCI_2', 'CCI_5', 'CCI_22', 'CMO_2', 'CMO_5',
       'CMO_22', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
       'MACD_6_30_9', 'MACDh_6_30_9', 'MACDs_6_30_9', 'MACD_24_52_9',
       'MACDh_24_52_9', 'MACDs_24_52_9', 'PPO_12_26_9', 'PPOh_12_26_9',
       'PPOs_12_26_9', 'PPO_24_52_9', 'PPOh_24_52_9', 'PPOs_24_52_9',
       'PVO_12_26_9', 'PVOh_12_26_9', 'PVOs_12_26_9', 'PVO_24_52_9',
       'PVOh_24_52_9', 'PVOs_24_52_9', 'MFI_2', 'MFI_5', 'RSI_2', 'RSI_5',
       'RSI_14', 'UO_5_15_30', 'WILLR_3', 'WILLR_5', 'WILLR_10', 'WILLR_20', 'K_9_3',
       'D_9_3', 'J_9_3', 'K_19_3', 'D_19_3', 'J_19_3', 'NATR_3', 'NATR_10', 'LOGRET_10', 
       'PCTRET_1', 'PCTRET_2', 'PCTRET_3', 'PCTRET_4', 'PCTRET_5', 'ZS_5',
       'ZS_14', 'RVI_5', 'RVI_14', 'rolling_money_3', 'rolling_money_5', 'rolling_money_10',
       'rolling_volume_3', 'rolling_volume_5', 'rolling_volume_10', 'pct_volatility',
       'rolling_pct_volatility_3', 'rolling_pct_volatility_5',
       'rolling_pct_volatility_10']).fillna(0.0)
        feature_df.columns = ['daily_%s' % i for i in feature_df.columns]

        feature_df['code'] = dayline_df['code']
        feature_df = feature_df.reset_index().set_index('code').reset_index()
        return feature_df
    
    ## 平均趋向指数
    try:
        adx_2 = (dayline_df.ta.adx(length=2) / 100).fillna(0.0)
        assert adx_2.columns.tolist() == ['ADX_2', 'DMP_2', 'DMN_2']
    except:
        adx_2 = pd.DataFrame(index=dayline_df.index, columns=['ADX_2', 'DMP_2', 'DMN_2']).fillna(0.0)
    try:
        adx_5 = (dayline_df.ta.adx(length=5) / 100).fillna(0.0)
        assert adx_5.columns.tolist() == ['ADX_5', 'DMP_5', 'DMN_5']
    except:
        adx_5 = pd.DataFrame(index=dayline_df.index, columns=['ADX_5', 'DMP_5', 'DMN_5']).fillna(0.0)
    try:
        adx_22 = (dayline_df.ta.adx(length=22) / 100).fillna(0.0)
        assert adx_22.columns.tolist() == ['ADX_22', 'DMP_22', 'DMN_22']
    except:
        adx_22 = pd.DataFrame(index=dayline_df.index, columns=['ADX_22', 'DMP_22', 'DMN_22']).fillna(0.0)
    
    ## 顺势指标
    try:
        cci_2 = (dayline_df.ta.cci(length=2) / 1000).to_frame().fillna(0.0).rename(columns={"CCI_2_0.015": "CCI_2"})
        assert cci_2.columns.tolist() == ['CCI_2']
    except:
        cci_2 = pd.DataFrame(index=dayline_df.index, columns=['CCI_2']).fillna(0.0)
    try:
        cci_5 = (dayline_df.ta.cci(length=5) / 1000).to_frame().fillna(0.0).rename(columns={"CCI_5_0.015": "CCI_5"})
        assert cci_5.columns.tolist() == ['CCI_5']
    except:
        cci_5 = pd.DataFrame(index=dayline_df.index, columns=['CCI_5']).fillna(0.0)
    try:
        cci_22 = (dayline_df.ta.cci(length=22) / 1000).to_frame().fillna(0.0).rename(columns={"CCI_22_0.015": "CCI_22"})
        assert cci_22.columns.tolist() == ['CCI_22']
    except:
        cci_22 = pd.DataFrame(index=dayline_df.index, columns=['CCI_22']).fillna(0.0)
        
    ## 钱德动量摆动指标
    try:
        cmo_2 = (dayline_df.ta.cmo(length=2) / 100).to_frame().fillna(0.0)
        assert cmo_2.columns.tolist() == ['CMO_2']
    except:
        cmo_2 = pd.DataFrame(index=dayline_df.index, columns=['CMO_2']).fillna(0.0)
    try:
        cmo_5 = (dayline_df.ta.cmo(length=5) / 100).to_frame().fillna(0.0)
        assert cmo_5.columns.tolist() == ['CMO_5']
    except:
        cmo_5 = pd.DataFrame(index=dayline_df.index, columns=['CMO_5']).fillna(0.0)
    try:
        cmo_22 = (dayline_df.ta.cmo(length=22) / 100).to_frame().fillna(0.0)
        assert cmo_22.columns.tolist() == ['CMO_22']
    except:
        cmo_22 = pd.DataFrame(index=dayline_df.index, columns=['CMO_22']).fillna(0.0)
        
    ## 指数平滑移动平均线 MACD
    try:
        macd_12_26_9 = dayline_df.ta.macd(12, 26, 9) 
        for k in macd_12_26_9:
            macd_12_26_9[k] = macd_12_26_9[k].div(dayline_df['close'].values) * 10
        macd_12_26_9 = macd_12_26_9.fillna(0.0)
        assert  macd_12_26_9.columns.tolist() == ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
    except:
        macd_12_26_9 = pd.DataFrame(index=dayline_df.index, columns=['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']).fillna(0.0)
        
    try:
        macd_6_30_9 = dayline_df.ta.macd(6, 30, 9) 
        for k in macd_6_30_9:
            macd_6_30_9[k] = macd_6_30_9[k].div(dayline_df['close'].values) * 10
        macd_6_30_9 = macd_6_30_9.fillna(0.0)
        assert  macd_6_30_9.columns.tolist() == ['MACD_6_30_9', 'MACDh_6_30_9', 'MACDs_6_30_9']
    except:
        macd_6_30_9 = pd.DataFrame(index=dayline_df.index, columns=['MACD_6_30_9', 'MACDh_6_30_9', 'MACDs_6_30_9']).fillna(0.0)
    
    try:
        macd_24_52_9 = dayline_df.ta.macd(24, 52, 9) 
        for k in macd_24_52_9:
            macd_24_52_9[k] = macd_24_52_9[k].div(dayline_df['close'].values) * 10
        macd_24_52_9 = macd_24_52_9.fillna(0.0)
        assert  macd_24_52_9.columns.tolist() == ['MACD_24_52_9', 'MACDh_24_52_9', 'MACDs_24_52_9']
    except:
        macd_24_52_9 = pd.DataFrame(index=dayline_df.index, columns=['MACD_24_52_9', 'MACDh_24_52_9', 'MACDs_24_52_9']).fillna(0.0)
        
    ## 指数平滑移动平均线 PPO
    try:
        ppo_12_26_9 = (dayline_df.ta.ppo(12, 26, 9) / 10).fillna(0.0)
        assert ppo_12_26_9.columns.tolist() == ['PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9']
    except:
        ppo_12_26_9 = pd.DataFrame(index=dayline_df.index, columns=['PPO_12_26_9', 'PPOh_12_26_9', 'PPOs_12_26_9']).fillna(0.0)
    try:
        ppo_24_52_9 = (dayline_df.ta.ppo(24, 52, 9) / 10).fillna(0.0)
        assert ppo_24_52_9.columns.tolist() == ['PPO_24_52_9', 'PPOh_24_52_9', 'PPOs_24_52_9']
    except:
        ppo_24_52_9 = pd.DataFrame(index=dayline_df.index, columns=['PPO_24_52_9', 'PPOh_24_52_9', 'PPOs_24_52_9']).fillna(0.0)
        
    try:
        pvo_12_26_9 = (dayline_df.ta.pvo(12, 26, 9) / 100).fillna(0.0)
        assert pvo_12_26_9.columns.tolist() == ['PVO_12_26_9', 'PVOh_12_26_9', 'PVOs_12_26_9']
    except:
        pvo_12_26_9 = pd.DataFrame(index=dayline_df.index, columns=['PVO_12_26_9', 'PVOh_12_26_9', 'PVOs_12_26_9']).fillna(0.0)
    try:
        pvo_24_52_9 = (dayline_df.ta.pvo(24, 52, 9) / 100).fillna(0.0)
        assert pvo_24_52_9.columns.tolist() == ['PVO_24_52_9', 'PVOh_24_52_9', 'PVOs_24_52_9']
    except:
        pvo_24_52_9 = pd.DataFrame(index=dayline_df.index, columns=['PVO_24_52_9', 'PVOh_24_52_9', 'PVOs_24_52_9']).fillna(0.0)
        
    try:
        mfi_2 = (dayline_df.ta.mfi(length=2) / 100).to_frame().fillna(0.5)
        assert mfi_2.columns.tolist() == ['MFI_2']
    except:
        mfi_2 = pd.DataFrame(index=dayline_df.index, columns=['MFI_2']).fillna(0.5)
    try:
        mfi_5 = (dayline_df.ta.mfi(length=5) / 100).to_frame().fillna(0.5)
        assert mfi_5.columns.tolist() == ['MFI_5']
    except:
        mfi_5 = pd.DataFrame(index=dayline_df.index, columns=['MFI_5']).fillna(0.5)    
    
    try:
        rsi_2 = (dayline_df.ta.rsi(length=2) / 100).to_frame().fillna(0.5)
        assert rsi_2.columns.tolist() == ['RSI_2']
    except:
        rsi_2 = pd.DataFrame(index=dayline_df.index, columns=['RSI_2']).fillna(0.5) 
    try:
        rsi_5 = (dayline_df.ta.rsi(length=5) / 100).to_frame().fillna(0.5)
        assert rsi_5.columns.tolist() == ['RSI_5']
    except:
        rsi_5 = pd.DataFrame(index=dayline_df.index, columns=['RSI_5']).fillna(0.5) 
    try:
        rsi_14 = (dayline_df.ta.rsi(length=14) / 100).to_frame().fillna(0.5)
        assert rsi_14.columns.tolist() == ['RSI_14']
    except:
        rsi_14 = pd.DataFrame(index=dayline_df.index, columns=['RSI_14']).fillna(0.5) 
   
    try:
        uo_5_15_30 = (dayline_df.ta.uo(5, 15, 30) / 100).to_frame().fillna(0.5)
        assert uo_5_15_30.columns.tolist() == ['UO_5_15_30']
    except:
        uo_5_15_30 = pd.DataFrame(index=dayline_df.index, columns=['UO_5_15_30']).fillna(0.0)
        
    try:
        willr_3 = (dayline_df.ta.willr(length=3) / 100).to_frame().fillna(-0.5)
        assert willr_3.columns.tolist() == ['WILLR_3']
    except:
        willr_3 = pd.DataFrame(index=dayline_df.index, columns=['WILLR_3']).fillna(-0.5) 
    try:
        willr_5 = (dayline_df.ta.willr(length=5) / 100).to_frame().fillna(-0.5)
        assert willr_5.columns.tolist() == ['WILLR_5']
    except:
        willr_5 = pd.DataFrame(index=dayline_df.index, columns=['WILLR_5']).fillna(-0.5) 
    try:
        willr_10 = (dayline_df.ta.willr(length=10) / 100).to_frame().fillna(-0.5)
        assert willr_10.columns.tolist() == ['WILLR_10']
    except:
        willr_10 = pd.DataFrame(index=dayline_df.index, columns=['WILLR_10']).fillna(-0.5) 
    try:
        willr_20 = (dayline_df.ta.willr(length=20) / 100).to_frame().fillna(-0.5)
        assert willr_20.columns.tolist() == ['WILLR_20']
    except:
        willr_20 = pd.DataFrame(index=dayline_df.index, columns=['WILLR_20']).fillna(-0.5) 
        
    try:
        kdj_9_3 = (dayline_df.ta.kdj(9, 3) / 100).fillna(0.5)
        assert kdj_9_3.columns.tolist() == ['K_9_3', 'D_9_3', 'J_9_3']
    except:
        kdj_9_3 = pd.DataFrame(index=dayline_df.index, columns=['K_9_3', 'D_9_3', 'J_9_3']).fillna(0.5)
    try:
        kdj_19_3 = (dayline_df.ta.kdj(19, 3) / 100).fillna(0.5)
        assert kdj_19_3.columns.tolist() == ['K_19_3', 'D_19_3', 'J_19_3']
    except:
        kdj_19_3 = pd.DataFrame(index=dayline_df.index, columns=['K_19_3', 'D_19_3', 'J_19_3']).fillna(0.5)
        
    try:
        natr_3 = (dayline_df.ta.natr(length=3) / 10).to_frame().fillna(0.5)
        assert natr_3.columns.tolist() == ['NATR_3']
    except:
        natr_3 = pd.DataFrame(index=dayline_df.index, columns=['NATR_3']).fillna(0.5) 
    try:
        natr_10 = (dayline_df.ta.natr(length=10) / 10).to_frame().fillna(0.5)
        assert natr_10.columns.tolist() == ['NATR_10']
    except:
        natr_10 = pd.DataFrame(index=dayline_df.index, columns=['NATR_10']).fillna(0.5)
    
    try:
        log_return_10 = (dayline_df.ta.log_return(length=10) * 10 ).clip(-10, 10).fillna(0.0).to_frame()
        assert log_return_10.columns.tolist() == ['LOGRET_10']
    except:
        log_return_10 = pd.DataFrame(index=dayline_df.index, columns=['LOGRET_10']).fillna(0.)
        
    try:
        percent_return_1 = (dayline_df.ta.percent_return(length=1)).to_frame().fillna(0.0)
        assert percent_return_1.columns.tolist() == ['PCTRET_1']
    except:
        percent_return_1 = pd.DataFrame(index=dayline_df.index, columns=['PCTRET_1']).fillna(0.0)
    try:
        percent_return_2 = (dayline_df.ta.percent_return(length=2)).to_frame().fillna(0.0) 
        assert percent_return_2.columns.tolist() == ['PCTRET_2']
    except:
        percent_return_2 = pd.DataFrame(index=dayline_df.index, columns=['PCTRET_2']).fillna(0.0)
    try:
        percent_return_3 = (dayline_df.ta.percent_return(length=3)).to_frame().fillna(0.0)
        assert percent_return_3.columns.tolist() == ['PCTRET_3']
    except:
        percent_return_3 = pd.DataFrame(index=dayline_df.index, columns=['PCTRET_3']).fillna(0.0)
    try:
        percent_return_4 = (dayline_df.ta.percent_return(length=4)).to_frame().fillna(0.0)
        assert percent_return_4.columns.tolist() == ['PCTRET_4']
    except:
        percent_return_4 = pd.DataFrame(index=dayline_df.index, columns=['PCTRET_4']).fillna(0.0)
    try:
        percent_return_5 = (dayline_df.ta.percent_return(length=5)).to_frame().fillna(0.0)
        assert percent_return_5.columns.tolist() == ['PCTRET_5']
    except:
        percent_return_5 = pd.DataFrame(index=dayline_df.index, columns=['PCTRET_5']).fillna(0.0)

    try:
        zscore_5 = (dayline_df.ta.zscore(length=5)).to_frame().fillna(0.0)
        assert zscore_5.columns.tolist() == ['ZS_5']
    except:
        zscore_5 = pd.DataFrame(index=dayline_df.index, columns=['ZS_5']).fillna(0.0)
    try:
        zscore_14 = (dayline_df.ta.zscore(length=14)).to_frame().fillna(0.0)
        assert zscore_14.columns.tolist() == ['ZS_14']
    except:
        zscore_14 = pd.DataFrame(index=dayline_df.index, columns=['ZS_14']).fillna(0.0)
    
    try:
        rvi_5 =  (dayline_df.ta.rvi(length=5) / 100).fillna(0.5).to_frame()
        assert rvi_5.columns.tolist() == ['RVI_5']
    except:
        rvi_5 = pd.DataFrame(index=dayline_df.index, columns=['RVI_5']).fillna(0.0)
    try:
        rvi_14 =  (dayline_df.ta.rvi(length=14) / 100).fillna(0.5).to_frame()
        assert rvi_14.columns.tolist() == ['RVI_14']
    except:
        rvi_14 = pd.DataFrame(index=dayline_df.index, columns=['RVI_14']).fillna(0.5)
    
    rolling_money_3 = ((np.log1p((dayline_df['money']).rolling(3, min_periods=1).mean()) - np.log1p((dayline_df['money']))).clip(-10, 10)).to_frame().rename(columns={'money': 'rolling_money_3'})
    rolling_money_5 = ((np.log1p((dayline_df['money']).rolling(5, min_periods=1).mean()) - np.log1p((dayline_df['money']))).clip(-10, 10)).to_frame().rename(columns={'money': 'rolling_money_5'})
    rolling_money_10 = ((np.log1p((dayline_df['money']).rolling(10, min_periods=1).mean()) - np.log1p((dayline_df['money']))).clip(-10, 10)).to_frame().rename(columns={'money': 'rolling_money_10'})
    rolling_volume_3 = ((np.log1p((dayline_df['volume']).rolling(3, min_periods=1).mean()) - np.log1p((dayline_df['volume']))).clip(-10, 10)).to_frame().rename(columns={'volume': 'rolling_volume_3'})
    rolling_volume_5 = ((np.log1p((dayline_df['volume']).rolling(5, min_periods=1).mean()) - np.log1p((dayline_df['volume']))).clip(-10, 10)).to_frame().rename(columns={'volume': 'rolling_volume_5'})
    rolling_volume_10 = ((np.log1p((dayline_df['volume']).rolling(10, min_periods=1).mean()) - np.log1p((dayline_df['volume']))).clip(-10, 10)).to_frame().rename(columns={'volume': 'rolling_volume_10'})
    pct_volatility = ((dayline_df['high'] - dayline_df['low'])  / dayline_df['close'] * 20).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'pct_volatility'})

    rolling_pct_volatility_3 = ((dayline_df['high'].rolling(3, min_periods=1).max() - dayline_df['low'].rolling(3, min_periods=1).min()) / dayline_df['close']* 20).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'rolling_pct_volatility_3'})
    rolling_pct_volatility_5 = ((dayline_df['high'].rolling(5, min_periods=1).max() - dayline_df['low'].rolling(5, min_periods=1).min()) / dayline_df['close'] * 20).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'rolling_pct_volatility_5'})
    rolling_pct_volatility_10 = ((dayline_df['high'].rolling(10, min_periods=1).max() - dayline_df['low'].rolling(10, min_periods=1).min()) / dayline_df['close'] * 20).clip(-10, 10).fillna(0.0).to_frame().rename(columns={0: 'rolling_pct_volatility_10'})

    
    feature_df = pd.concat([adx_2, adx_5, adx_22, cci_2, cci_5, cci_22, cmo_2, cmo_5, cmo_22, macd_12_26_9, macd_6_30_9, macd_24_52_9, ppo_12_26_9, ppo_24_52_9, pvo_12_26_9, pvo_24_52_9, mfi_2, mfi_5, rsi_2, rsi_5, rsi_14, uo_5_15_30, willr_3, willr_5, willr_10, willr_20, kdj_9_3, kdj_19_3, natr_3, natr_10, log_return_10, percent_return_1, percent_return_2, percent_return_3, percent_return_4, percent_return_5, zscore_5, zscore_14, rvi_5, rvi_14, rolling_money_3, rolling_money_5, rolling_money_10, rolling_volume_3, rolling_volume_5, rolling_volume_10, pct_volatility, rolling_pct_volatility_3, rolling_pct_volatility_5, rolling_pct_volatility_10], axis=1)
    feature_df.columns = ['daily_%s' % i for i in feature_df.columns]

    feature_df['code'] = dayline_df['code']
    feature_df = feature_df.reset_index().set_index('code').reset_index()
    feature_df.iloc[:, 2:] = feature_df.iloc[:, 2:].clip(-10, 10).astype(np.float32)

    return feature_df

```

## A股因子数据和标签生成

### 标签生成
本次使用申万行业分类作为建模目标，预测当天及次日的开盘、平均、收盘收益，即：
- 今开/昨收：今日开盘相对昨天收盘的涨跌百分比(动量效应)
- 今收/昨收：今日收盘相对昨天收盘的涨跌百分比(轮动效应)
- 今均/昨收：今日平均价格相对昨天收盘的涨跌百分比(轮动效应)
- 今收/今开：今日收盘相对今日开盘的涨跌百分比(日内趋势)
- 今均/今开：今日平均价格相对今日开盘的涨跌百分比(日内趋势)
- 明开/今开：明日开盘价格相对于今日开盘的涨跌百分比(短线效应)
- 明收/今开：明日收盘价格相对于今日开盘的涨跌百分比(短线轮动)
- 明均/今开：明日平均价格相对于今日开盘的涨跌百分比(短线轮动)
- 明开/今均：明日开盘价格相对于今日平均的涨跌百分比(短线轮动)
- 明均/今均：明日平均价格相对于今日平均的涨跌百分比(短线轮动)

标签生成的过程可参考上篇`利用pandas_ta自动提取技术面特征`,一个例子为：
```python
# 日内：平均相对于昨收的涨跌百分比(开盘9:15时间截断)
sector_feature_intraday_avg_close = sector_feature_grouped.apply(lambda row: ((row['high'] + row['low']) / (2 * row.shift(1)['close']) - 1) * 100).clip(-10, 10).reset_index().rename(columns={0: 'intraday_avg_close'}).set_index(['trade_day', 'code'])

# 日间：第二天收盘相对于前一天开盘的涨跌百分比(开盘9:15时间截断)
sector_feature_daily_close_open = sector_feature_grouped.apply(lambda row: (row.shift(-1)['close'] / row['open'] - 1) * 100).clip(-10, 10).reset_index().rename(columns={0: 'daily_close_open'}).set_index(['trade_day', 'code'])
```

最终，我们得到类似这样的标签数据：
![](/img/trade_label.png)

### 技术因子生成
这里的技术因子同上美股市场，不再赘述。选择的标的为申万行业指数、沪深300成分股及其指数、中证500成分股及其指数，总共约1100多只。

提取完美股和A股的技术因子后，需要将两者(cn_trade_day、us_trade_day)合并，与标签(trade_label_df)一起形成一个数据集。合并的规则是，以A股标签为基准，选择最近一个日期的A/美因子数据，考虑到部分标的数据可能存在的停牌、未上市、退市等特殊情况，还需要对技术因子生成一个掩码，排除掉这些无效数据：

```python
data = [] 
mask = []
for ind, row in tqdm(trade_label_df.iterrows(), total=len(trade_label_df)):
    fea_mask = [0] * (1119 + 607)
    trade_day = datetime.date(ind // 10000, (ind % 10000) // 100, ind % 100)
    
    cn_last_day = [i for i in cn_trade_day if i < trade_day]
    cn_fea = cn_df[cn_df.trade_day == cn_last_day[-1]]
    cn_fea['code'] = cn_fea['code'] - 1
    cn_fea_data = [0 ] * 74 * 1119
    for row2 in cn_fea.values:
        cn_fea_data[74 * row2[0]: 74 * (row2[0] + 1)] = row2[2:]
        fea_mask[row2[0]] = 1
        
    us_last_day = [i for i in us_trade_day if i < trade_day]
    us_fea = us_df[us_df.trade_day == us_last_day[-1]]
    us_fea['code'] = us_fea['code'] - 1300
    us_fea_data = [0 ] * 74 * 607
    for row2 in us_fea.values:
        us_fea_data[74 * row2[0]: 74 * (row2[0] + 1)] = row2[2:]
        fea_mask[row2[0] + 1119] = 1
    data.append(cn_fea_data + us_fea_data)
    mask.append(fea_mask)

X = np.array(data, dtype=np.float32)
mask = np.array(mask, dtype=np.int32)
```

## AI建模 

训练集为20220624之前的因子数据及其标签，验证集为20220624之后的15天数据，评估模型的实战效果。

### 数据读取
使用Pytorch的Dataset读取提取的因子数据：

```python
class TechDataset(Dataset):
    def __init__(self, label=None, mask=None, feature=None, seq_len=1726, output_size=223):
        self.label = label
        self.mask = mask
        self.feature = feature
        self.seq_len = seq_len
        self.output_size = output_size
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        label = torch.tensor(self.label[idx].reshape((self.output_size, 10)), dtype=torch.float32)
        input_mask = torch.tensor(self.mask[idx], dtype=torch.bool)
        input_techs = torch.tensor(self.feature[idx].reshape((self.seq_len, 74)), dtype=torch.float32)
        
        return label, input_mask, input_techs
    
X = np.load('trade_X.npy')
y =  np.load('trade_y.npy')
mask = np.load('trade_y_mask.npy')

train_X, valid_X, train_y, valid_y, train_mask_X, valid_mask_X = X[:-15], X[-15:], y[:-15], y[-15:], mask[:-15], mask[-15:]
    
train_dataset = TechDataset(seq_len=1726, output_size=223, label=train_y, mask=train_mask_X, feature=train_X)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
valid_dataset = TechDataset(seq_len=1726, output_size=223, label=valid_y, mask=valid_mask_X, feature=valid_X)
valid_dataloader = DataLoader(valid_dataset, batch_size=15, shuffle=False, num_workers=4)
```

### 结构设计

首先通过一个GLU单元将技术因子数据维度标准到64维，然后使用Transformer对因子数据建模，输出一个`Batch * Seq_len * Dim`的向量，然后通过`TechPredictor`部分进行池化、分类，最终得到`Batch * 223 * 10`的结果，223为标签种类，10为每类收益指标。

```python

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate=None,
                 activation = None):
        
        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
            
        self.W4 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = nn.Linear(self.input_size, self.hidden_layer_size)
        
        if self.activation_name:
            self.activation = getattr(nn, self.activation_name)()
            
        self.sigmoid = nn.Sigmoid()
            
        self.init_weights()
            
    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)
            
    def forward(self, x):
        if self.dropout_rate:
            x = self.dropout(x)
      
        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)
            
        return output


class TechEncoder(nn.Module):
    def __init__(self, depth=1, dim=64, group_size=64, query_key_dim=32, attn_dropout=0.2):
        super().__init__()
        self.dim = dim
        self.tech_emb = GatedLinearUnit(74, dim)
        self.transformer_layers = nn.ModuleList([FLASH(dim=dim, group_size=group_size, query_key_dim=query_key_dim, causal=False, dropout=attn_dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, input_techs, attention_mask):
        input_feature = self.tech_emb(input_techs) # Batch * seq_len  * dim
        for flash in self.transformer_layers:
            input_feature = flash(input_feature, mask=attention_mask)
        output = self.norm(input_feature) # Batch  * seq_len * dim
        
        return output, attention_mask
    
class TechPredictor(nn.Module):
    def __init__(self, depth=1, dim=64, seq_len=1726, output_size=223, class_num=10):
        super(TechPredictor, self).__init__()
        self.tech_encoder = TechEncoder(depth=depth, dim=dim)
        stride = np.floor(seq_len / output_size).astype(int)
        kernel_size =  seq_len - (output_size - 1) * stride
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.classifier = nn.Linear(in_features=dim, out_features=class_num)
    def forward(self, input_techs, attention_mask):
        tech_feature, attention_mask = self.tech_encoder(input_techs, attention_mask) 
        tech_feature = tech_feature * attention_mask.unsqueeze(-1)
        output = self.max_pooling(tech_feature.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() # Batch  * 223 * 128
        output = self.classifier(output)  # Batch  * 223 * 10
        return output

```

### 损失函数

直接使用均方误差损失函数优化模型，同时也计算出pearson相关系数供参考：

```python
def calc_loss(y_true, y_pred):
    y_true = y_true.reshape(-1, 10)
    y_pred = y_pred.reshape(-1, 10)
    y_true_label = y_true - y_true.mean(dim=0, keepdim=True)
    y_pred_label =  y_pred - y_pred.mean(dim=0, keepdim=True)
    loss = F.mse_loss(y_pred, y_true)
    pearson = torch.cosine_similarity(y_true_label ,y_pred_label, dim=0, eps=1e-6)
    return loss, pearson
```

### 训练主流程

迭代30次，每次训练完后评估效果：
```python
    model = TechPredictor(depth=1, dim=128, seq_len=1726, output_size=223, class_num=10)

    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=2e-5)
    step = 0
    
    for epoch in range(0, 30):
        model.train()
        for ind, batch in enumerate(train_dataloader):
            label, input_mask, input_techs = batch
            logit = model(input_techs.to(device), input_mask.to(device))
            loss, pearson = calc_loss(label.to(device), logit)
                  
            writer.add_scalars('train', {'loss':loss.item()}, step)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            step += 1
        model.eval()
        with torch.no_grad():
            loss_valid = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
            loss_mean = []
            for ind, batch in enumerate(valid_dataloader):
                label, input_mask, input_techs = batch
                logit = model(input_techs.to(device), input_mask.to(device))
                loss, pearson = calc_loss(label.to(device), logit)
                for ind, v in  enumerate(pearson.cpu().tolist()):
                    loss_valid[ind].append(v)
                loss_mean.append(loss.item())         
        print("Epoch: %d, loss: %.4f" % (epoch, np.mean(loss_mean)))
        torch.save(model.state_dict(), model_dir + "/tech_model/TechPredictor2_%d.torch" % epoch)      
```

## 结果分析

将验证集上的预测结果取出，基准数据沪深300、中证500、创业板指的指标均取`明均/今均`,预测指标限定在次日收益('明均/今均', '明均/今开', '明收/今均', '明开/今开', '明开/今均')中，`Top5收益`表示取Top5的预测行业对应的平均收益，`Top5预估值`表示AI模型预测的均值，Top10类似。

|     日期 |   沪深300 |   中证500 |   创业板指 |   Top5收益 |   Top5预估值 |   Top10收益 |   Top10预估值 |
|---------:|----------:|----------:|-----------:|-----------:|-------------:|------------:|--------------:|
| 20220624 |     1.603 |     1.238 |      1.737 |      4.416 |        0.578 |       2.686 |         0.568 |
| 20220627 |     0.205 |     0.461 |     -1.069 |      2.752 |        0.515 |       1.88  |         0.5   |
| 20220628 |     0.133 |    -0.212 |     -0.22  |      1.277 |        0.614 |       0.479 |         0.601 |
| 20220629 |     0.179 |     0.054 |     -0.046 |      0.406 |        0.526 |       0.19  |         0.521 |
| 20220630 |     0.193 |     0.207 |     -0.285 |      1.504 |        0.632 |       1.074 |         0.612 |
| 20220701 |    -0.295 |     0.191 |     -0.151 |     -0.218 |        0.858 |       0.207 |         0.817 |
| 20220704 |     0.515 |     0.42  |      1.164 |      1.628 |        0.485 |       2.107 |         0.472 |
| 20220705 |    -1.026 |    -0.983 |     -0.231 |      0.218 |        0.485 |      -0.1   |         0.472 |
| 20220706 |    -0.329 |     0.048 |      0.182 |     -0.496 |        0.53  |      -0.755 |         0.515 |
| 20220707 |     0.654 |     0.247 |      0.946 |      0.906 |        0.471 |       1.522 |         0.45  |
| 20220708 |    -1.905 |    -1.8   |     -2.652 |     -1.338 |        0.228 |      -1.445 |         0.22  |
| 20220711 |    -0.799 |    -0.609 |     -1.36  |     -0.425 |        0.327 |      -0.922 |         0.31  |
| 20220712 |    -0.468 |    -0.713 |     -0.636 |     -1.397 |        0.494 |      -1.267 |         0.456 |
| 20220713 |     0.214 |     0.79  |      2.588 |      0.666 |        0.583 |       1.286 |         0.57  |
| 20220714 |    -0.595 |    -0.586 |      0.319 |     -1.321 |        0.421 |      -1.239 |         0.414 |

从结果上看`Top5收益`整体是优于基准指数的，平均收益达到0.5以上，如果做一次过滤，取预估值大于0.5，那么平均收益可达1.2以上，可以起到一定的择时、选股效果。

以6月24号为例，AI模型选择出的Top5行业以及买卖时间策略，收益高达4个点以上：

|   trade_day | sector         | metric    |   actual |   prediction |
|------------:|:---------------|:----------|---------:|-------------:|
|    20220624 | 申万多业态零售 | 明均/今开 |    1.6   |        0.586 |
|    20220624 | 申万酒店       | 明均/今开 |    7.362 |        0.584 |
|    20220624 | 申万旅游综合   | 明均/今开 |    8.8   |        0.576 |
|    20220624 | 申万百货       | 明均/今开 |    1.812 |        0.573 |
|    20220624 | 申万超市       | 明均/今开 |    2.508 |        0.572 |

#### 推荐阅读

- [利用pandas_ta自动提取技术面特征](https://mp.weixin.qq.com/s/PPduk4xPcix9USW9HmUpHw)
- [基于新闻事件Bert序列建模的行业涨跌预测](https://mp.weixin.qq.com/s/CJxhVB6m2-DINp1mGNL4Bw)
---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。

![](/img/qrcode.jpg)