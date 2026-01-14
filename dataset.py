import re
import jieba
import logging
# 引入英文分词库，例如 NLTK 或者直接使用空格分词
# 如果没有安装 NLTK，请先 pip install nltk
from nltk.tokenize import word_tokenize
import csv
from torch.utils.data import Dataset

jieba.setLogLevel(logging.INFO)

# 中文正则表达式（保留中英数字）
chinese_regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
# 英文正则表达式（保留英文字母和数字，移除其他符号）
english_regex = re.compile(r'[^a-zA-Z0-9\s]')


def chinese_word_cut(text):
    """
    中文分词函数，使用jieba。
    """
    text = chinese_regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]

def english_word_tokenize(text):
    """
    英文分词函数，使用NLTK的word_tokenize。
    """
    text = english_regex.sub(' ', text) # 清理英文文本中的非字母数字字符
    # 将文本转换为小写，这是英文NLP的常见做法
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.strip()] # 移除空字符串

class CustomTextDataset(Dataset):
    # ... (CustomTextDataset 的其余部分保持不变)
    def __init__(self, data_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer # 这里的tokenizer现在可以是chinese_word_cut或english_word_tokenize
        self.vocab = None
        self.label_vocab = None
        self.raw_texts = []
        self.raw_labels = []

        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    label = row[1]
                    text = row[2]
                    self.raw_texts.append(text)
                    self.raw_labels.append(label)
        
        self.tokenized_texts = [self.tokenizer(text) for text in self.raw_texts]

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        if self.vocab is None or self.label_vocab is None:
            raise RuntimeError("词汇表必须在迭代数据集之前构建并赋值给数据集实例。")

        text_tokens = self.tokenized_texts[idx]
        text_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in text_tokens]

        label = self.raw_labels[idx]
        label_id = self.label_vocab.get(label, self.label_vocab.get('<unk>', 0))

        return {'text': text_ids, 'label': label_id}

def get_dataset(path, tokenizer):
    """
    加载训练集和开发集。
    tokenizer 参数现在可以是 chinese_word_cut 或 english_word_tokenize。
    """
    train_dataset = CustomTextDataset(f"{path}/train.tsv", tokenizer)
    dev_dataset = CustomTextDataset(f"{path}/test.tsv", tokenizer)
    return train_dataset, dev_dataset
