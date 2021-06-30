import torch
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import numpy as np
from utils import *



def load_datasets(args):  # 读文件，把adj、feature转成tensor,（adj, feature, tag）一起放到data里，再输入到Data_Dataset初始化

    # with open('data/mr/ind.mr.x_adj', 'rb')
    # print(len(a))
    # dev_size = 800

    with open("data/mr/ind.mr.{}".format('y'), 'rb') as f:  # 看训练集的大小
        y = pkl.load(f)
    # print(len(y))  # 6398
    # input('leny')

    names_train = ['allx_adj', 'allx_embed', 'ally']  # 训练和验证集
    objects = []
    for i in range(len(names_train)):
        with open("data/mr/ind.mr.{}".format(names_train[i]), 'rb') as f:
            objects.append(pkl.load(f))

    alladj, allembed, ally = tuple(objects)
    train_adj = []
    train_embed = []
    for i in range(len(ally)):  # 训练和验证集
        # print(np.array(allembed[i]).shape[1])
        # input('alladj[i]')
        a_i = alladj[i].toarray()
        embed_i = np.array(allembed[i])
        train_adj.append(a_i)
        train_embed.append(embed_i)

    # train_adj = np.array(train_adj)
    # train_embed = np.array(train_embed)
    train_y = np.array(ally)

    train_data = []
    dev_data = []
    for i in range(0, len(y)):  # train
        dicti = {'adj': train_adj[i], 'embed': train_embed[i], 'y': train_y[i]}
        train_data.append(dicti)

    for i in range(len(y), len(train_adj)):  # eval
        dicti = {'adj': train_adj[i], 'embed': train_embed[i], 'y': train_y[i]}
        dev_data.append(dicti)



    names_test = ['tx_adj', 'tx_embed', 'ty']
    objects = []
    for i in range(len(names_test)):
        with open("data/mr/ind.mr.{}".format(names_test[i]), 'rb') as f:
            objects.append(pkl.load(f))

    adj, embed, y = tuple(objects)
    test_adj = []
    test_embed = []
    for i in range(len(y)):
        a_i = adj[i].toarray()
        embed_i = np.array(embed[i])
        test_adj.append(a_i)
        test_embed.append(embed_i)

    test_y = np.array(y)

    test_data = []
    for i in range(0, len(test_adj)):
        dicti = {'adj': test_adj[i], 'embed': test_embed[i], 'y': test_y[i]}
        test_data.append(dicti)

    train_dataset = Data_Dataset(train_data, args)
    dev_dataset = Data_Dataset(dev_data, args)
    test_dataset = Data_Dataset(test_data, args)

    return train_dataset, dev_dataset, test_dataset


class Data_Dataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        items = d['adj'], d['embed'], d['y']
        items_tensor = tuple(t for t in items)
        # print(items_tensor)
        return items_tensor

    # def convert_features(self):
    #     '''
    #     Convert sentence, aspects, pos_tags, dependency_tags to ids.
    #     '''
    #     for i in range(len(self.data)):
    #         self.convert_features_bert(i)
    #         self.data[i]['text_len'] = len(self.data[i]['input_ids'])-2



def pad_batch(batch):

    adjs, features, label = zip(*batch)
    label = torch.tensor(label)

    adjs = list(adjs)
    features = list(features)

    # pad adj
    max_length = max([a.shape[0] for a in adjs])
    mask = np.zeros((len(adjs), max_length, 1))  # mask for padding

    for i in range(len(adjs)):
        adj_normalized = normalize_adj(adjs[i])  # no self-loop
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adjs[i].shape[0], :] = 1.
        adjs[i] = adj_normalized

    # adjs = torch.Tensor(np.array(list(adjs)))
    adjs = torch.Tensor(adjs)



    # pad features
    max_length = max([len(f) for f in features])

    for i in range(len(features)):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]  # padding for each epoch
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    features = torch.Tensor(features)

    mask = torch.tensor(mask)

    return adjs, mask, features, label


def pad_batch_train(batch):

    adjs, features, label = zip(*batch)
    label = torch.tensor(label)

    # print(type(features))
    # # print(adjs)
    # input('aaaaa')
    adjs = list(adjs)
    features = list(features)

    # pad adj
    # max_length = max([a.shape[0] for a in adjs])
    max_length = 44
    mask = np.zeros((len(adjs), max_length, 1))  # mask for padding

    for i in range(len(adjs)):
        adj_normalized = normalize_adj(adjs[i])  # no self-loop
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adjs[i].shape[0], :] = 1.
        adjs[i] = adj_normalized

    adjs = torch.Tensor(adjs)

    max_length = 44

    for i in range(len(features)):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]  # padding for each epoch
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    # features = np.array(list(features))
    features = torch.Tensor(features)
    # return np.array(list(features))

    mask = torch.tensor(mask)

    return adjs, mask, features, label


def pad_batch_eval(batch):

    adjs, features, label = zip(*batch)
    label = torch.tensor(label)

    adjs = list(adjs)
    features = list(features)
    max_length = 40
    mask = np.zeros((len(adjs), max_length, 1))  # mask for padding

    for i in range(len(adjs)):
        adj_normalized = normalize_adj(adjs[i])  # no self-loop
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adjs[i].shape[0], :] = 1.
        adjs[i] = adj_normalized

    adjs = torch.Tensor(adjs)

    max_length = 40

    for i in range(len(features)):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]  # padding for each epoch
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    features = torch.Tensor(features)

    mask = torch.tensor(mask)

    return adjs, mask, features, label

















