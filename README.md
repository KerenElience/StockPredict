# 说明
当前工作仅仅使用TFT对未来5日收益值进行预测

# 安装
1. 下载source code
2. 执行`pip install -m requirement.txt`，pytorch自行根据显卡驱动版本安装。
3. 额外需要单独安装ta-lib，以及sqlite3.
4. 执行`python precomputer_lmdb.py`
5. 运行`stock.ipynb`

## 历史数据
Akshare抓取数据（日线），使用splite3进行存储在本地，避免重复抓取导致数据获取失败。
时线运行时获取当天，判断交易点（待完成）。

- 输入：时间窗口长度内的多个数据指标：基础指标`open`, `close`, `low`, `high`, `amount`，技术指标暂时添加额外的17种
- 标签：预测标签为选取时间窗口外紧接着的未来5日log平均收益值。


### 训练，验证及测试数据生成
根据`CNNDataset`获得的数据，还需要进一步进行时间切片，依据限定日期范围获取所有数据，而后将数据随机按某个固定时间窗口采样。采样数据需要按行进行Z-score标准化。
分别生成train Data, valid Data, test Data.

### CNN神经网络构建
- base model: 使用简单MLP进行预测。
- fc层：单值，softmax。

### LSTM构建底模
- LSTM: 2 layer,
- fc层：单值，tanh。

### 损失函数
- MSE
- IC值，自定义





