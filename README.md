## TextCNN Pytorch实现 文本分类（中文或英文）
## 论文
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

## 数据准备
数据集自备，统一格式为tsv，第一列为id，第二列为label，第三列为text
可使用class.py作为参考统一格式


## 用法
训练全部的参数都在conf/config.py中，可根据需要修改

## 词表
词表根据数据集生成，可单独下载比较成熟的词表

## 训练
```bash
python3 main.py
```

## 预测
```bash
python3 predict.py
```