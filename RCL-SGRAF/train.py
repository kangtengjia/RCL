"""
# Pytorch implementation for AAAI2021 paper from
# https://arxiv.org/pdf/2101.01368.
# "Similarity Reasoning and Filtration for Image-Text Matching"
# Haiwen Diao, Ying Zhang, Lin Ma, Huchuan Lu
#
# Writen by Haiwen Diao, 2020
"""

import os
import time
import shutil

import torch
from transformers import BertTokenizer
import numpy

import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores

import logging
import math
from roma import is_roma_dataset
from roma_evaluation import text_to_scene_metrics
import tensorboard_logger as tb_logger

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    opt = opts.parse_opt()
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    if opt.text_enc_type == 'bert':
        vocab = BertTokenizer.from_pretrained(opt.bert_path, local_files_only=True)
        opt.vocab_size = len(vocab.vocab)
    else:
        vocab_file = 'nr3d_vocab.json' if opt.data_name == 'nr3d' else 'my_data_vocab.json'
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, vocab_file))
        vocab.add_word('<mask>')
        opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SGRAF(opt)

    # Train the Model
    best_r1 = 0
    start_epoch = 0
    if len(opt.resume) > 0:
        checkpoint = torch.load(opt.resume, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        # Preserve command-line options. Historical checkpoints embed the
        # original ``opt`` object, which otherwise discards a requested lower
        # learning rate and the new output directory.
        if opt.resume_reset_epoch:
            start_epoch = 0
        else:
            start_epoch = checkpoint['epoch']
            best_r1 = checkpoint.get('best_r1', 0)
        logging.info('Loaded %s; start_epoch=%d, best_R@1=%g, learning_rate=%g',
                     opt.resume, start_epoch, best_r1, opt.learning_rate)
    # r_sum = validate(opt, val_loader, model)
    epochs_without_improvement = 0
    for epoch in range(start_epoch, opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        r1 = validate(opt, val_loader, model)

        # Select checkpoints and early-stop solely by text-to-scene R@1.
        is_best = r1 > best_r1
        if is_best:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        best_r1 = max(r1, best_r1)

        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_r1': best_r1,
            'selection_metric': 'R@1',
            'opt': opt,
            'Eiters': model.Eiters,
        # }, is_best, filename='{}_{}_checkpoint_{}_{}_{}.pth.tar'.format(opt.data_name, opt.module_name, opt.noise_rate, opt.margin, epoch), prefix=opt.model_name + '/')
        }, is_best, filename='{}_{}_checkpoint_{}_{}_{}.pth.tar'.format(opt.data_name, opt.module_name, opt.noise_rate, opt.loss, epoch), prefix=opt.model_name + '/')

        if opt.early_stop_patience > 0 and epochs_without_improvement >= opt.early_stop_patience:
            logging.info(
                'Early stopping at epoch %d: validation R@1 did not improve for %d epochs.',
                epoch + 1, opt.early_stop_patience)
            break


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    # validate(opt, val_loader, model)
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)
    if is_roma_dataset(opt.data_name):
        scene_rows = [val_loader.dataset.scene_indices.index(scene) for scene in dict.fromkeys(val_loader.dataset.scene_indices)]
        sims = shard_attn_scores(model, img_embs[scene_rows], cap_embs, cap_lens, opt, shard_size=100)
        metrics = text_to_scene_metrics(sims.T, val_loader.dataset.scene_indices)
        logging.info('Text to scene: %s', metrics)
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                tb_logger.log_value('roma/' + name, value, step=model.Eiters)
        return metrics['R@1']
    img_div = 1 if 'cc152k' in opt.data_name else 5 #int(val_loader.dataset.im_div)
    # clear duplicate 5*images and keep 1*images
    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), img_div)])

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims, img_div=img_div)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims, img_div=img_div)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r_sum', r_sum, step=model.Eiters)

    return r_sum


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    opt = state['opt']
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                print('====================Saving the Best Model========================')
                shutil.copyfile(prefix + filename, prefix + opt.best_model_filename)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Update each optimizer group without changing its configured base LR."""
    if opt.lr_schedule == 'cosine_restart':
        if opt.lr_cycle_epochs < 2:
            raise ValueError('--lr_cycle_epochs must be at least 2 for cosine_restart')
        cycle_epoch = epoch % opt.lr_cycle_epochs
        phase = cycle_epoch / float(opt.lr_cycle_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
        for param_group in optimizer.param_groups:
            base_lr = param_group.get('initial_lr', opt.learning_rate)
            lr = opt.lr_min + (base_lr - opt.lr_min) * cosine
            param_group['lr'] = lr
        return

    for param_group in optimizer.param_groups:
        base_lr = param_group.get('initial_lr', opt.learning_rate)
        lr = base_lr * (0.1 ** (epoch // opt.lr_update))
        if param_group.get('group_name') == 'bert' and opt.bert_warmup_epochs > 0:
            lr *= min(1.0, float(epoch + 1) / opt.bert_warmup_epochs)
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
