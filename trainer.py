import logging
import os
import random
from utils import compute_metrics
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import torch.optim as optim
# from datasets import my_collate, my_collate_elmo, my_collate_pure_bert, my_collate_bert
from dataset import pad_batch
# from dataset import pad_batch_train, pad_batch_eval
from transformers import AdamW
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def train(args, train_dataset, model, test_dataset, eval_dataset):
    train_sampler = RandomSampler(train_dataset)
    collate_fn1 = pad_batch  # pad a batch
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=collate_fn1)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=1e-5)  # weight_decay=1e-5 if for L2 Reg
    criterion = torch.nn.CrossEntropyLoss()

    epochs = args.epochs
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    dev_best_loss = float('inf')
    flag = False

    for epoch in range(epochs):
        logger.info('**********************  epoch: %d  **********************', epoch)
        model.train()

        for data in train_dataloader:
            adj, mask, emb, y = data
            adj = adj.float().to(args.device)
            mask = mask.float().to(args.device)
            emb = emb.float().to(args.device)
            y = y.float().to(args.device)
            optimizer.zero_grad()

            y = torch.argmax(y, dim=1)

            output = model(emb, adj, mask)


            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                acc, f1, eval_loss = evaluate(args, eval_dataset, model)

                if eval_loss < dev_best_loss:
                    dev_best_loss = eval_loss
                    torch.save(model.state_dict(), args.model_save_path)
                    improve = '*'
                    last_improve = global_step
                else:
                    improve = ''

                logger.info('**********************  evaluate  **********************')
                logger.info("global_step: %d, train_loss: %f, eval_acc: %f, eval_f1: %f, eval_loss: %f, improve: %s",
                            global_step, (tr_loss - logging_loss) / args.logging_steps, acc, f1, eval_loss, improve)


                logging_loss = tr_loss

                tacc, tf1, teval_loss = test(args, test_dataset, model)
                logger.info('**********************  test  **********************')
                logger.info("test_acc: %f, test_f1: %f, test_loss: %f", tacc, tf1, teval_loss)


                model.train()


            if global_step - last_improve > args.require_improvement:
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break


def test(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn2 = pad_batch
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.batch_size, collate_fn=collate_fn2)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=1e-5)  # weight_decay=1e-5 if for L2 Reg
    criterion = torch.nn.CrossEntropyLoss()

    eval_loss = 0.0
    nb_eval_steps = 0

    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            adj, mask, emb, y = batch
            adj = adj.float().to(args.device)
            mask = mask.float().to(args.device)
            emb = emb.float().to(args.device)
            y = y.float().to(args.device)

            y = torch.argmax(y, dim=1)
            output = model(emb, adj, mask)

            loss = criterion(output, y)
            eval_loss += loss.item()

        nb_eval_steps += 1


        if preds is None:
            preds = output.detach().cpu().numpy()
            out_label_ids = y.detach().cpu().numpy()
        else:
            preds = np.append(preds, output.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, y.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    acc, f1 = compute_metrics(preds, out_label_ids)

    return acc, f1, eval_loss


def evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn2 = pad_batch
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.batch_size, collate_fn=collate_fn2)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=1e-5)  # weight_decay=1e-5 if for L2 Reg
    criterion = torch.nn.CrossEntropyLoss()

    eval_loss = 0.0
    nb_eval_steps = 0

    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            adj, mask, emb, y = batch
            adj = adj.float().to(args.device)
            mask = mask.float().to(args.device)
            emb = emb.float().to(args.device)
            y = y.float().to(args.device)

            y = torch.argmax(y, dim=1)

            output = model(emb, adj, mask)
            loss = criterion(output, y)
            eval_loss += loss.item()
        nb_eval_steps += 1


        if preds is None:
            preds = output.detach().cpu().numpy()
            out_label_ids = y.detach().cpu().numpy()
        else:
            preds = np.append(preds, output.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, y.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    acc, f1 = compute_metrics(preds, out_label_ids)
    return acc, f1, eval_loss






