---
title: 基于新闻事件Bert序列建模的行业涨跌预测
date: 2022-07-08 10:00:00
tags: 
    - 新闻事件
    - 量化交易
    - 量化策略
categories:
    - 量化
---

> 俗话说"女怕嫁错郎，男怕入错行"，选择往往比努力更重要，在股票市场也不例外，一个好的标的往往能让我们瞬间躺赢，而一旦踩雷则万劫不复。金融市场从长期来看均是受经济增长因素驱动，短期走势则是受资金的流动影响，而新闻事件和各种消息对市场价格的影响非常快速，影响交易情绪。今天我们利用先进的深度学习技术单独分析新闻事件序列对行业指数的短期影响，和读者一起探讨利用AI做行业选股的可行性。

## 行业板块数据处理

读取数据库中的行业板块日线数据，包含3类：
- 大盘指数：上证指数、上证50、沪深300、中证500等
- CSRC1行业分类：如CSRC1采矿业、CSRC1建筑业等
- CSRC2行业分类：如CSRC2农副食品加工业、CSRC2开采辅助活动等

上述行业板块数据合计108个，然后提取相关的指标：

```python
# 读取数据库中的数据
sector_data = client2.execute("select code, trade_day, sector_name, sector_type, open, close, high, low, volume, amount from xtquant.sector_1d")
# 转换成DataFrame
sector_df = pd.DataFrame(sector_data, columns=['code', 'trade_day', 'sector_name', 'sector_type', 'open', 'close', 'high', 'low', 'volume', 'amount'])
sector_df = sector_df.drop_duplicates(subset=['trade_day', 'code'])

# 每个指数分别处理
sector_feature_grouped = sector_df.sort_values(by=['trade_day', 'code']).set_index('trade_day').groupby(['code'])

# 日内： 开盘相对昨收的涨跌百分比(开盘9:15时间截断)
sector_feature_intraday_open_close = sector_feature_grouped.apply(lambda row: (row['open'] / row.shift(1)['close'] - 1) * 100).clip(-10, 10).reset_index().rename(columns={0: 'intraday_open_close'}).set_index(['trade_day', 'code'])

# 日间：第二天开盘相对于前一天开盘的涨跌百分比(开盘9:15时间截断)
sector_feature_daily_open_open = sector_feature_grouped.apply(lambda row: (row.shift(-1)['open'] / row['open'] - 1) * 100).clip(-10, 10).reset_index().rename(columns={'open': 'daily_open_open'}).set_index(['trade_day', 'code'])

```

指标共10个，基准分为当天和第二天：
- intraday_open_close：开盘相对昨收的涨跌百分比
- intraday_close_close：收盘相对昨收的涨跌百分比
- intraday_avg_close：平均相对于昨收的涨跌百分比
- intraday_close_open：收盘相对开盘的涨跌百分比
- intraday_avg_open：平均相对于开盘的涨跌百分比
- daily_open_open：第二天开盘相对于前一天开盘的涨跌百分比
- daily_close_open：第二天收盘相对于前一天开盘的涨跌百分比
- daily_avg_open：第二天平均相对于前一天开盘的涨跌百分比
- daily_open_avg：第二天开盘相对于前一天平均价格的涨跌百分比
- daily_avg_avg：第二天平均相对于前一天平均的涨跌百分比

其中当天的指标主要优化模型的拟合能力，即过去发生的事对当前的影响，而第二天指标优化模型的预测能力，即过去发生的事对未来的影响。这里重点关注第二天指标，因为A股为T+1交易，这关系到能否利用已发生的事件赚取未来的收益。

数据样例如下, 时间戳为索引，列共`108*10 = 1080`个：

![](/img/label_data.png)

## 新闻数据采集和Bert微调

这里主要使用了两个数据集：财联社快讯和华盛通快讯的爬虫数据，时间维度从2016年至今，进行清洗后，按照时间排序。数据如下表所示：

|   timestamp | content                                                                            |
|------------:|:-----------------------------------------------------------------------------------|
|  1460337101 | 一季度新能源乘用车销量3.95万 同比翻番                                              |
|  1515081541 | 英国商务大臣克拉克：努力深化与欧盟27国的贸易联系。                                 |
|  1535487757 | 据外媒：美国参议院确定克拉里达成为美联储的副主席。                                 |
|  1611292392 | 吉林通化第二轮检测已检31.6万人 51例阳性                                            |
|  1617094987 | 洪汇新材：上调氯乙烯-醋酸乙烯共聚树脂二元系列产品的国内销售报价                    |
|  1617662023 | 公募“大年”FOF大丰收 规模业绩实现双增长                                             |
|  1628664753 | 中兴通讯AH股走低，目前均跌近8%；其他电讯设备股中，国微控股跌超6%，京信通信跌1.5%。 |
|  1629450813 | 香港交易所计划推出全新MSCI中国A50互联互通指数期货合约。（港交所）                  |
|  1632368553 | 韩媒：韩国预计在中秋假日后新冠感染病例将上升                                       |
|  1633867053 | 中工国际：前三季净利预增63.52%-86.88%                                              |


由于BERT模型使用的数据与这里的数据有较大差异，因此可以使用掩码语言模型(MLM)进行进一步的微调，[sentence-bert](https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/MLM/train_mlm.py)提供了一个训练脚本，这里可直接拿来使用，只需使用content行的文本即可，每行一个新闻，划分好开发集和测试集，便可进行微调改进。


## 涨跌预测回归模型构建

首先导入相关的包，定义相关的路径：

``` python
# 设置使用的显卡
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# pytorch 程序包
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# transformers预训练模型包
from transformers import BertModel, BertTokenizer, BertConfig
# 长序列建模包
from flash_pytorch import FLASH

import numpy as np
import pandas as pd
import datetime, json, random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
# 农历日期组件
from lunardate import LunarDate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 项目路径
base_dir = 'path_to_your_bert'
# 模型存储路径
model_dir = base_dir + 'model/'
# 预训练孟子模型
bert_dir = model_dir + 'mengzi-bert-base-mlm/'
# 输入数据集
input_dir = base_dir + 'input/'
# tensorboard日志路径
log_dir =  base_dir + 'log/'

```

### 数据读取

构造数据的主要思路是取当天交易日的9:15前的4096条新闻文本，经过Bert分词编码后，对对应的时间特征一起用来训练，标签及其未缺失的掩码用来优化指导AI模型。

```python

class NewsDataset(Dataset):
    def __init__(self, seq_len=4096, txt_len=64, label_df=None, mask_df=None, news_df=None, tokenizer=None):
        self.label_df = label_df
        self.mask_df = mask_df
        self.news_df = news_df
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.txt_len = txt_len 
    def __len__(self):
        return len(self.label_df)
    def parse_news_time(self, x, base):
        interval = int(np.clip(np.log1p(base - x), 0, 20))
        ts = datetime.datetime.fromtimestamp(x)
        month = ts.month
        day = ts.day
        weekday = ts.weekday()
        luna_date = LunarDate.fromSolarDate(ts.year, month, day)
        luna_month = luna_date.month
        luna_day = luna_date.day
        return [month, day, weekday, interval, luna_month, luna_day]
    def __getitem__(self, idx):
        ts = self.label_df.iloc[idx]
        label = torch.tensor(ts.tolist(), dtype=torch.float32).reshape((108, 10))
        mask = torch.tensor(self.mask_df.iloc[idx].astype(int).tolist(), dtype=torch.bool)
        news = news_df[news_df.timestamp<=ts.name].iloc[-(self.seq_len + 1024):].sort_values(by=['timestamp'], ascending=False)
        news = pd.concat([news.iloc[:1024], news.iloc[1024:].sample(self.seq_len - 1024).sort_values(by=['timestamp'], ascending=False)], axis=0)
        input_ids = tokenizer(news['content'].tolist(), max_length=self.txt_len, truncation=True, padding=True, return_tensors="pt")['input_ids']
        input_times = news["timestamp"].apply(lambda x: self.parse_news_time(x, ts.name)).tolist()
        input_times = torch.tensor(input_times, dtype=torch.long)
        
        return label, mask, input_ids, input_times
    
```

### 时间特征编码

这里对时间6个特征进行编码，然后合并到一起，组成和文本编码相同的维度(BERT的最后输出为768)，作为Transformer模型的位置编码特征。

```python
class TimeEncoder(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.month_emb = nn.Embedding(13, dim // 8)
        self.day_emb = nn.Embedding(32, dim // 8 )
        self.lunar_month_emb = nn.Embedding(13, dim // 8)
        self.lunar_day_emb = nn.Embedding(32, dim // 8 )
        self.weekday_emb = nn.Embedding(8, dim // 4)
        self.interval_emb = nn.Embedding(20, dim // 4)
    def forward(self, times):
        month_emb = self.month_emb(times[:, 0])
        day_emb = self.day_emb(times[:, 1])
        weekday_emb = self.weekday_emb(times[:, 2])
        interval_emb = self.interval_emb(times[:, 3])
        lunar_month_emb = self.lunar_month_emb(times[:, 4])
        lunar_day_emb = self.lunar_day_emb(times[:, 5])
        time_embedding = torch.cat([month_emb, day_emb, weekday_emb, interval_emb, lunar_month_emb, lunar_day_emb], dim=1)
        return time_embedding
```

### 新闻BERT编码

这里使用了澜舟科技开源的[孟子预训练模型](https://github.com/Langboat/Mengzi)对单个新闻文本进行编码，并使用了`MLM`任务进行了进一步的微调。最终可以得到一个`Batch  * Seq_len * Dim` 的文本向量表示。需要注意的是，我们对孟子模型的最后三层设置了可再训练，会消耗较大显存，如果资源不足，可设置不再优化。对于多个新闻序列，我们这里使用了[Flash-Transformer](https://github.com/lucidrains/FLASH-pytorch)，一个长序列建模工具，最终得到新闻序列的向量表示。

```python
class NewsEncoder(nn.Module):
    def __init__(self, depth=3, dim=768, seq_len=4096, txt_len=64, group_size=256, query_key_dim=128, attn_dropout=0.2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.txt_len = txt_len
        if os.path.exists(bert_dir):
            self.bert_encoder = BertModel.from_pretrained(bert_dir)
        else:
            self.bert_encoder = BertModel(BertConfig.from_json_file(model_dir + 'bert_config.json')) 
        for name, param in self.bert_encoder.named_parameters():
            param.requires_grad = False
            for n in ['layer.10', 'layer.11', 'pooler.dense']:
                if n in name:
                    param.requires_grad = True
        self.time_encoder = TimeEncoder(dim=768)
        self.transformer_layers = nn.ModuleList([FLASH(dim=dim, group_size = group_size, query_key_dim = query_key_dim, causal=False, dropout = attn_dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, input_ids, input_times):
        '''
        input_ids: [Batch * seq_len * txt_len]
        input_times: [Batch * seq_len * time_len]
        '''
        input_ids = input_ids.reshape((-1, input_ids.shape[-1])) # (Batch * seq_len) * txt_len
        input_times = input_times.reshape((-1, 6)) # (Batch * seq_len) * 6
        
        attention_mask = input_ids.ne(0).long().to(input_ids)
        
        bert_output = self.bert_encoder(input_ids,  attention_mask=attention_mask)  # (Batch * seq_len) * txt_len * dim
        time_output = self.time_encoder(input_times) # (Batch * seq_len) * dim
        news_feature = time_output + bert_output.pooler_output # (Batch * seq_len)  * dim
        news_feature = news_feature.reshape((-1, self.seq_len, self.dim)) # Batch  * seq_len * dim
        
        for flash in self.transformer_layers:
            news_feature = flash(news_feature)
        output = self.norm(news_feature) # Batch  * seq_len * dim
        
        return output
   
```

### 指标预测

由于将一段时间内多个新闻汇集到一起进行预测，因此需要将上述新闻向量表示进行进一步地加工，首先进行平均池化操作，然后对接一个分类器，形成`Batch  * 108 * 10`维度的预测值，即代表着108个板块中，10个指标的预测情况。

```python 
class IndustryPredictor(nn.Module):
    def __init__(self, depth=2, dim=768, seq_len=4096, class_num=10):
        super(IndustryPredictor, self).__init__()
        self.news_encoder = NewsEncoder(depth=depth, dim=dim, seq_len=seq_len)
        self.avg_pooling = nn.AdaptiveAvgPool1d(108)
        self.classifier = nn.Linear(in_features=dim, out_features=class_num)
    def forward(self, input_ids, input_times):
        news_feature = self.news_encoder(input_ids, input_times) 
        output = self.avg_pooling(news_feature.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() # Batch  * 108 * 768
        output = self.classifier(output)  # Batch  * 108 * 10
        return output

```

### 损失函数

虽然我们可以直接用MSE均方误差损失函数对预测结果进行梯度优化，但考虑到部分指标并不是一直存在的，例如科创50指数早期并不存在，也就没有group truth，会造成指标的不一致，因此，这里我们使用掩码的方式将标签和对应的预测值遮蔽，避免造成干扰。这里也同时计算了pearson相关系数指标，在验证的时候可以直观地看到预测值和真实值的相关程度。
```python
def calc_loss(y_true, y_pred, mask):
    mask2 = mask.squeeze(0).reshape((108, 1)).repeat(1, 10).unsqueeze(0)
    y_true = torch.masked_select(y_true.unsqueeze(0), mask2).reshape(-1, 10)
    y_pred = torch.masked_select(y_pred.unsqueeze(0), mask2).reshape(-1, 10)
    y_true_label = y_true - y_true.mean(dim=0, keepdim=True)
    y_pred_label =  y_pred - y_pred.mean(dim=0, keepdim=True)
    loss = F.mse_loss(y_pred, y_true)
    pearson = torch.cosine_similarity(y_true_label ,y_pred_label, dim=0, eps=1e-6)
    return loss, pearson
```

### 主训练过程

`FLASH-Transformer`深度为3， 序列长度为4096，使用AdamW优化器，学习率为0.00005

```python
    writer = SummaryWriter(log_dir + '/tensorbord/') 
    model = IndustryPredictor(depth=3, dim=768, seq_len=4096)
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=5e-5)
    step = 0
    
    for epoch in range(10):
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        model.train()
        for ind, batch in enumerate(train_dataloader):
            label, mask, input_ids, input_times = batch
            logit = model(input_ids.to(device), input_times.to(device))
            loss, pearson = calc_loss(label.to(device), logit, mask.to(device))
                  
            writer.add_scalars('train', {'loss':loss.item()}, step)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            step += 1
        torch.save(model.state_dict(), model_dir + "/news_model/IndustryPredictor_%d.torch" % epoch)
```

## 分析预测结果

### 查看训练过程的损失函数变化曲线

![](/img/loss.png)

从图中可以看出,训练过程中的损失函数曲线十分曲折，就是使用0.99的阈值平滑，噪声也很大。因此，这并不是一个容易学习的任务，预测的结果具有很高的随机性。

### 查看验证集上的预测值和实际值的相关性系数

我们在验证集上对108个板块，10个单项指标，合计1080个总指标的预测值与真实值之间的相关系数(`sector_corr_df["corr"].describe()`)进行统计,可发现尽管总体上相关程度不高，但也有一些较高的单项。

|       |         corr |
|:------|-------------:|
| count | 1080         |
| mean  |   -0.030514  |
| std   |    0.274666  |
| min   |   -0.649692  |
| 25%   |   -0.221297  |
| 50%   |   -0.0553582 |
| 75%   |    0.136897  |
| max   |    0.864099  

我们详细看看那些预测较高(`sector_corr_df[sector_corr_df["corr"] > 0.6]`))的指标：

| sector                                  | indicator           |     corr |
|:----------------------------------------|:--------------------|---------:|
| CSRC2农副食品加工业                     | intraday_open_close | 0.780031 |
| CSRC2食品制造业                         | intraday_open_close | 0.84407  |
| CSRC2酒、饮料和精制茶制造业             | intraday_open_close | 0.731186 |
| CSRC2皮革、毛皮、羽毛及其制品和制鞋业   | intraday_open_close | 0.635645 |
| CSRC2木材加工及木、竹、藤、棕、草制品业 | intraday_open_close | 0.694802 |
| CSRC2家具制造业                         | intraday_open_close | 0.825688 |
| CSRC2造纸及纸制品业                     | intraday_open_close | 0.755133 |
| CSRC2橡胶和塑料制品业                   | intraday_open_close | 0.636808 |
| CSRC2金属制品业                         | intraday_open_close | 0.838386 |
| CSRC2通用设备制造业                     | intraday_open_close | 0.811593 |
| CSRC2专用设备制造业                     | intraday_open_close | 0.828883 |
| CSRC2汽车制造业                         | intraday_open_close | 0.728715 |
| CSRC2计算机、通信和其他电子设备制造业   | intraday_open_close | 0.681775 |
| CSRC2仪器仪表制造业                     | intraday_open_close | 0.76271  |
| CSRC2仪器仪表制造业                     | intraday_avg_close  | 0.600887 |
| CSRC2废弃资源综合利用业                 | intraday_open_close | 0.617559 |
| CSRC2电力、热力生产和供应业             | intraday_open_close | 0.679018 |
| CSRC2燃气生产和供应业                   | intraday_open_close | 0.708935 |
| CSRC2土木工程建筑业                     | intraday_open_close | 0.666156 |
| CSRC2建筑装饰和其他建筑业               | intraday_open_close | 0.864099 |
| CSRC2建筑装饰和其他建筑业               | intraday_avg_close  | 0.709512 |
| CSRC2批发业                             | intraday_open_close | 0.605899 |
| CSRC2批发业                             | intraday_avg_close  | 0.612417 |
| CSRC2电信、广播电视和卫星传输服务       | intraday_open_close | 0.649785 |
| CSRC2互联网和相关服务                   | intraday_open_close | 0.683816 |

可发现日内相关指标的系数较高，而日间较低，这与预期相符，毕竟第二天不可控的因素更多。

### 具体日期预测结果分析

我们以`2022-07-01`日的预测结果回看准确性。将预测的结果倒序，排除日内指标，可看出推荐的Top5隔天平均收益达到1个百分点，整体还是可观的。

| trade_date   | sector                                        | metric           |   actual |   prediction |
|:-------------|:----------------------------------------------|:-----------------|---------:|-------------:|
| 2022-07-01   | CSRC2铁路运输业                               | daily_close_open |   -1.282 |        0.921 |
| 2022-07-01   | CSRC2林业                                     | daily_close_open |    2.471 |        0.899 |
| 2022-07-01   | CSRC2有色金属矿采选业                         | daily_close_open |    3.187 |        0.895 |
| 2022-07-01   | CSRC2零售业                                   | daily_close_open |   -0.882 |        0.886 |
| 2022-07-01   | CSRC2农业                                     | daily_close_open |    1.727 |        0.875 |
| 2022-07-01   | CSRC2黑色金属矿采选业                         | daily_close_open |    3.532 |        0.871 |
| 2022-07-01   | CSRC2化学原料及化学制品制造业                 | daily_close_open |    1.628 |        0.867 |
| 2022-07-01   | CSRC2电气机械及器材制造业                     | daily_close_open |    0.953 |        0.861 |
| 2022-07-01   | CSRC2有色金属冶炼及压延加工业                 | daily_close_open |    1.94  |        0.85  |
| 2022-07-01   | CSRC2畜牧业                                   | daily_close_open |    9.056 |        0.84  |
| 2022-07-01   | CSRC2水上运输业                               | daily_close_open |   -0.427 |        0.835 |
| 2022-07-01   | CSRC2金属制品业                               | daily_close_open |    1.102 |        0.811 |
| 2022-07-01   | CSRC2航空运输业                               | daily_close_open |   -3.904 |        0.801 |
| 2022-07-01   | CSRC2医药制造业                               | daily_close_open |    2.501 |        0.8   |
| 2022-07-01   | CSRC2石油加工、炼焦及核燃料加工业             | daily_close_open |    0.381 |        0.797 |
| 2022-07-01   | CSRC2铁路、船舶、航空航天和其它运输设备制造业 | daily_close_open |    0.922 |        0.789 |
| 2022-07-01   | CSRC2道路运输业                               | daily_close_open |   -1.089 |        0.783 |
| 2022-07-01   | CSRC2道路运输业                               | daily_avg_open   |   -1.441 |        0.783 |
| 2022-07-01   | CSRC2通用设备制造业                           | daily_close_open |    0.666 |        0.782 |
| 2022-07-01   | CSRC2非金属矿采选业                           | daily_close_open |    2.067 |        0.778 |

## 结论

利用序列模型对新闻事件数据进行拟合、预测，虽然不是非常精确、整体相关性不是很明显，但在一些突发事件中，还是可以获得一些超额收益的。大家如有观点看法，欢迎后台留言讨论！

---

欢迎关注我的公众号“**量化实战**”，原创技术文章第一时间推送。
![](/img/qrcode.jpg)
