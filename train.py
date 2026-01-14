import os
import sys
import torch
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args, writer):
    if args.device.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning.lr)
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.learning.epochs + 1):
        total_epoch_loss = 0.0
        total_epoch_corrects = 0
        total_epoch_samples = 0
        steps = 0
        for batch in train_iter:
            feature, target = batch['text'], batch['label']
            print(feature.size())
            feature.data.t_()#, target.data.sub_(1)
            if args.device.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            print(feature.size())
            s
            logits = model(feature)
            # print(logits, target)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            predicted = torch.max(logits, 1)[1].view(target.size())
            current_batch_corrects = (predicted == target).sum().item() # 转换为 Python 整数
            current_batch_size = target.size(0)
            total_epoch_loss += loss.item() * current_batch_size # 损失乘以样本数，以便后续加权平均
            total_epoch_corrects += current_batch_corrects
            total_epoch_samples += current_batch_size
            # if steps % args.log_interval == 0:
            #     corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
            #     train_acc = 100.0 * corrects / args.batch_size
            #     print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, loss.item(), train_acc,corrects, args.batch_size))
        if total_epoch_samples > 0:
            avg_epoch_loss = total_epoch_loss / total_epoch_samples
            avg_epoch_acc = 100.0 * total_epoch_corrects / total_epoch_samples
            
            print(
                '\nEpoch[{}] - Avg Loss: {:.6f}  Avg Acc: {:.4f}%({}/{})'.format(
                    epoch, avg_epoch_loss, avg_epoch_acc, total_epoch_corrects, total_epoch_samples
                )
            )
            writer.add_scalar('Loss/train', avg_epoch_loss, global_step=epoch)
            writer.add_scalar('Accuracy/train', avg_epoch_acc, global_step=epoch)
        else:
            print(f'\nEpoch[{epoch}] - No samples processed for this epoch.')
        if epoch % args.learning.test_interval == 0:
            dev_acc = eval(dev_iter, model, args, writer, epoch)
            model.train()
            if dev_acc > best_acc:
                best_acc = dev_acc
                last_step = steps
                if args.learning.save_best:
                    print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                    save(model, args.learning.save_dir, 'best', steps)
            else:
                if steps - last_step >= args.learning.early_stopping:
                    print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.learning.early_stopping, best_acc))
                    raise KeyboardInterrupt


def eval(data_iter, model, args, writer, global_step):
    model.eval()
    corrects, avg_loss = 0, 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_iter:
            feature, target = batch['text'], batch['label']
            feature.data.t_()#, target.data.sub_(1)
            if args.device.cuda:
                feature, target = feature.cuda(), target.cuda()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            avg_loss += loss.item()# * feature.size(0)
            total_samples += feature.size(0)
            corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
    # size = len(data_iter.dataset)
    # avg_loss /= len(data_iter) #有修改
    final_avg_loss = avg_loss / total_samples 
    accuracy = 100.0 * corrects / total_samples
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       total_samples))
    writer.add_scalar('Loss/validation', final_avg_loss, global_step=global_step)
    writer.add_scalar('Accuracy/validation', accuracy, global_step=global_step)
    return accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
