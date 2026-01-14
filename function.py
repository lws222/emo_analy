import yaml
from types import SimpleNamespace # 用于将字典转换为可点访问的对象
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np # 用于加载词向量
from functools import partial # 用于绑定 collate_fn 参数
import pickle
import os
import dataset
def save_vocabs(word_vocab, label_vocab, save_dir="vocabs"):
    """
    保存词汇表到指定目录。

    Args:
        word_vocab (dict): 词语到ID的映射。
        label_vocab (dict): 标签到ID的映射。
        save_dir (str): 保存词汇表的目录。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    word_vocab_path = os.path.join(save_dir, "word_vocab.pkl")
    label_vocab_path = os.path.join(save_dir, "label_vocab.pkl")

    with open(word_vocab_path, 'wb') as f:
        pickle.dump(word_vocab, f)
    print(f"词语词汇表已保存到: {word_vocab_path}")

    with open(label_vocab_path, 'wb') as f:
        pickle.dump(label_vocab, f)
    print(f"标签词汇表已保存到: {label_vocab_path}")

def load_vocabs(load_dir="vocabs"):
    """
    从指定目录加载词汇表。

    Args:
        load_dir (str): 词汇表所在的目录。

    Returns:
        tuple: (word_vocab, label_vocab)
    """
    word_vocab_path = os.path.join(load_dir, "word_vocab.pkl")
    label_vocab_path = os.path.join(load_dir, "label_vocab.pkl")

    if not os.path.exists(word_vocab_path) or not os.path.exists(label_vocab_path):
        raise FileNotFoundError(f"词汇表文件未找到。请检查目录: {load_dir}")

    with open(word_vocab_path, 'rb') as f:
        word_vocab = pickle.load(f)
    print(f"词语词汇表已从: {word_vocab_path} 加载")

    with open(label_vocab_path, 'rb') as f:
        label_vocab = pickle.load(f)
    print(f"标签词汇表已从: {label_vocab_path} 加载")

    return word_vocab, label_vocab
def load_config(config_path):
    """
    加载并解析 YAML 配置文件。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 将嵌套的字典转换为 SimpleNamespace 对象，以便通过点号访问
    def dict_to_simplenamespace(d):
        if not isinstance(d, dict):
            return d
        # 对字典中的每个值递归调用此函数
        return SimpleNamespace(**{k: dict_to_simplenamespace(v) for k, v in d.items()})

    return dict_to_simplenamespace(config_dict)

def load_word_vectors(model_name, model_path, word_vocab):
    """
    从文件中加载预训练词向量（例如 GloVe, Word2Vec 格式）。
    假设文件格式为: 词 维度1 维度2 ... 维度N
    只加载存在于 word_vocab 中的词的向量。

    Args:
        model_name (str): 词向量模型的名称（在此函数中主要用于信息输出）。
        model_path (str): 预训练词向量文件的路径。
        word_vocab (dict): 词汇表，将词映射到整数ID。

    Returns:
        embedding_matrix (torch.Tensor): 包含预训练词向量的 Tensor。
        embedding_dim (int): 词向量的维度。
    """
    word_to_vec = {}
    embedding_dim = None
    
    # 将词汇表转换为集合，以便高效查找
    vocab_words_set = set(word_vocab.keys())

    with open(model_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: # 跳过空行
                continue
            word = parts[0]
            # 仅处理在我们的词汇表中出现的词
            if word in vocab_words_set: 
                try:
                    vector = np.array(parts[1:], dtype=np.float32)
                    if embedding_dim is None:
                        embedding_dim = len(vector)
                    elif embedding_dim != len(vector):
                        print(f"警告: 词 '{word}' 的词向量维度不匹配。预期 {embedding_dim}，实际 {len(vector)}。已跳过。")
                        continue # 如果维度不匹配，则跳过该词
                    word_to_vec[word] = vector
                except ValueError:
                    print(f"警告: 无法解析词 '{word}' 的词向量。已跳过。")
                    continue

    if embedding_dim is None:
        raise ValueError("无法从预训练词向量文件中确定词向量维度。请检查文件内容。")

    # 创建一个嵌入矩阵
    # 对于在预训练文件中未找到的词，其向量将保持为零（或可以进行随机初始化）
    embedding_matrix = np.zeros((len(word_vocab), embedding_dim), dtype=np.float32)
    
    # 填充预训练词向量
    for word, idx in word_vocab.items():
        if word in word_to_vec:
            embedding_matrix[idx] = word_to_vec[word]
        # else: 对于未找到的词，保持为零，或者根据您的模型需求进行其他初始化

    return torch.from_numpy(embedding_matrix), embedding_dim


def build_vocab(datasets, min_freq=1):
    """
    从 CustomTextDataset 实例列表中构建词汇表和标签词汇表。

    Args:
        datasets (list): CustomTextDataset 对象的列表。
        min_freq (int): 词语被包含在词汇表中的最小频率。

    Returns:
        word_vocab (dict): 将词映射到整数ID的字典。
        label_vocab (dict): 将标签映射到整数ID的字典。
    """
    token_counter = Counter()
    label_counter = Counter()

    for ds in datasets:
        # 使用数据集中的已分词文本和原始标签来构建计数器
        for tokens in ds.tokenized_texts:
            token_counter.update(tokens)
        
        label_counter.update(ds.raw_labels)

    # 创建词汇表
    word_vocab = {'<pad>': 0, '<unk>': 1} # 保留 0 用于填充，1 用于未知词
    idx = 2
    for word, count in token_counter.most_common():
        if count >= min_freq:
            word_vocab[word] = idx
            idx += 1

    # 创建标签词汇表
    label_vocab = {}
    idx = 0
    for label, count in label_counter.most_common():
        label_vocab[label] = idx
        idx += 1
    
    return word_vocab, label_vocab

def collate_fn(batch):
    """
    DataLoader 的 collate_fn 函数，用于对批次中的序列进行填充（padding）。

    Args:
        batch (list): 一个字典列表，每个字典包含 'text' (整数ID列表)
                      和 'label' (整数ID)。

    Returns:
        padded_texts (torch.Tensor): 填充后的文本序列。
        label_ids (torch.Tensor): 标签ID。
    """
    # 分离文本和标签
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]

    # 将批次中的所有序列填充到当前批次的最大长度
    max_len = max(len(t) for t in texts)
    padded_texts = torch.zeros(len(texts), max_len, dtype=torch.long) # 初始化为0（<pad>的ID）
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = torch.tensor(text, dtype=torch.long)

    # 将标签转换为 Tensor
    label_ids = torch.tensor(labels, dtype=torch.long)

    return {'text':padded_texts, 'label':label_ids}


def load_data_and_create_dataloaders(args):
    """
    加载原始数据集，构建词汇表，加载预训练词向量，并创建 DataLoader 实例。
    """
    print('正在加载原始数据集...')
    # 调用 dataset.py 中的 get_dataset 函数来获取 CustomTextDataset 实例
    train_dataset_obj, dev_dataset_obj = dataset.get_dataset(args.data.data_path, dataset.english_word_tokenize if args.data.tokenizer == 'english' else dataset.chinese_word_cut)

    print('正在构建词汇表...')
    word_vocab, label_vocab = build_vocab([train_dataset_obj, dev_dataset_obj])
    save_vocabs(word_vocab, label_vocab, save_dir=args.data.vocabs_dir)
    # 将构建好的词汇表赋值给数据集实例，以便 __getitem__ 方法进行数值化
    train_dataset_obj.vocab = word_vocab
    train_dataset_obj.label_vocab = label_vocab
    dev_dataset_obj.vocab = word_vocab
    dev_dataset_obj.label_vocab = label_vocab
    # 更新 args 中的词汇表信息
    args.model.vocabulary_size = len(word_vocab)
    args.model.class_num = len(label_vocab)

    # 如果需要，加载预训练词向量
    if args.model.static and args.model.pretrained_name and args.model.pretrained_path:
        print('正在加载预训练词向量...')
        embedding_matrix, embedding_dim = load_word_vectors(args.model.pretrained_name, args.model.pretrained_path, word_vocab)
        args.model.embedding_dim = embedding_dim
        args.model.vectors = embedding_matrix
    else:
        # 如果不使用预训练词向量，或者路径/名称未提供，则设定一个默认的嵌入维度
        # 模型的嵌入层将进行随机初始化
        if not args.model.static: # 如果 static 为 False，表示不使用预训练，则随机初始化
            args.model.embedding_dim = 100 # 示例默认维度
        else: # 如果 static 为 True 但没有提供预训练路径，则发出警告并仍使用默认维度
            print("警告: args.static 为 True 但未提供 pretrained_path/name 或未找到。嵌入层将随机初始化。")
            # args.model.embedding_dim = 100 # 默认维度
            args.model.vectors = None # 确保 args.vectors 为 None，指示模型进行随机初始化

    # 创建 DataLoader 实例
    train_dataloader = DataLoader(
        train_dataset_obj,
        batch_size=args.learning.batch_size,
        shuffle=True,
        collate_fn=collate_fn, # 使用自定义的 collate_fn 进行填充
        # num_workers=4, # 在实际应用中可以考虑增加 num_workers 以提高数据加载速度
    )

    # 对于开发集 DataLoader，原代码使用整个开发集作为一个批次
    dev_dataloader = DataLoader(
        dev_dataset_obj,
        batch_size=args.learning.batch_size, # 将整个开发集作为一个批次加载
        shuffle=False, # 开发集通常不需要打乱
        collate_fn=collate_fn,
    )

    return train_dataloader, dev_dataloader, word_vocab, label_vocab