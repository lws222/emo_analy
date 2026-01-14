import torch
import torch.nn as nn
import torch.optim as optim
from function import *
import model
import dataset
if __name__ == '__main__':

    config_file_path = 'conf/config.yaml'
    model_path = 'snapshot/best_steps_980.pt' #保存的模型pt文件地址
    args = load_config(config_file_path)
    loaded_word_vocab, loaded_label_vocab = load_vocabs(load_dir="en_vocabs") # 保存的词表文件夹地址
    args.model.class_num = 2 #类别数
    args.model.vocabulary_size = len(loaded_word_vocab)
    args.device.cuda = 0
    # 解析 filter_sizes
    args.model.filter_sizes = [int(size) for size in args.model.filter_sizes.split(',')]
    text_cnn = model.TextCNN(args.model)
    loaded_state_dict = torch.load(model_path, weights_only=False)
    text_cnn.load_state_dict(loaded_state_dict)
    text_cnn.to(f'cuda:{args.device.cuda}').eval()
    input_data = "I've seen this story before but my kids haven't. Boy with troubled past joins military, faces his past, falls in love and becomes a man. The mentor this time is played perfectly by Kevin Costner; An ordinary man with common everyday problems who lives an extraordinary conviction, to save lives. After losing his team he takes a teaching position training the next generation of heroes. The young troubled recruit is played by Kutcher. While his scenes with the local love interest are a tad stiff and don't generate enough heat to melt butter, he compliments Costner well. I never really understood Sela Ward as the neglected wife and felt she should of wanted Costner to quit out of concern for his safety as opposed to her selfish needs. But her presence on screen is a pleasure. The two unaccredited stars of this movie are the Coast Guard and the Sea. Both powerful forces which should not be taken for granted in real life or this movie. The movie has some slow spots and could have used the wasted 15 minutes to strengthen the character relationships. But it still works. The rescue scenes are intense and well filmed and edited to provide maximum impact. This movie earns the audience applause. And the applause of my two sons."
    text_tokens = dataset.english_word_tokenize(input_data)
    text_ids = [loaded_word_vocab.get(token, loaded_word_vocab['<unk>']) for token in text_tokens]
    feature= torch.tensor(text_ids).data.t_().unsqueeze(0).to(f'cuda:{args.device.cuda}')

    with torch.no_grad():
        logits = text_cnn(feature)
    predicted = torch.max(logits, 1)[1]
    print('识别结果为：','pos' if predicted==1 else 'neg')
