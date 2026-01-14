
import torch
import torch
import time
import model
import train
from torch.utils.tensorboard import SummaryWriter
import os
from function import *

if __name__ == '__main__':

    # --- 主程序执行 ---
        
    config_file_path = 'conf/config.yaml'
    args = load_config(config_file_path)
    print('正在加载数据...')
    # load_data_and_create_dataloaders 函数将返回 DataLoader 和构建好的词汇表
    train_iter, dev_iter, text_vocab, label_vocab = load_data_and_create_dataloaders(args)
    # 根据加载的数据更新 args 参数
    # args.vocabulary_size, args.class_num, args.embedding_dim, args.vectors 已经在 load_data_and_create_dataloaders 中设置

    if args.model.multichannel:
        args.model.static = True
        args.model.non_static = True

    # 检查 CUDA 是否可用
    args.device.cuda = args.device.id != -1 and torch.cuda.is_available()
    # 解析 filter_sizes
    args.model.filter_sizes = [int(size) for size in args.model.filter_sizes.split(',')]

    print(f"词汇表大小: {args.model.vocabulary_size}")
    print(f"类别数量: {args.model.class_num}")
    print(f"嵌入维度: {args.model.embedding_dim}")
    print(f"CUDA 可用: {args.device.cuda}")
    print(f"label_vocab: {label_vocab}")
    print('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        if attr in {'vectors'}:
            continue
        print('\t{}={}'.format(attr.upper(), value))


    text_cnn = model.TextCNN(args.model)
    if args.learning.snapshot:
        print('\nLoading model from {}...\n'.format(args.learning.snapshot))
        text_cnn.load_state_dict(torch.load(args.learning.snapshot))
    log_dir = os.path.join("runs")
    writer = SummaryWriter(log_dir)
    if args.device.cuda:
        torch.cuda.set_device(args.device.id)
        text_cnn = text_cnn.cuda()
    try:
        train.train(train_iter, dev_iter, text_cnn, args, writer)
    except KeyboardInterrupt:
        print('Exiting from training early')
