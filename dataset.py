import torch
from torch.utils.data import DataLoader, Dataset
import pickle as pkl
import numpy as np
from utils import *
from torch.nn.utils.rnn import pad_sequence



def load_datasets(args):  # 读文件，把adj、feature转成tensor,（adj, feature, tag）一起放到data里，再输入到Data_Dataset初始化

    # with open('data/mr/ind.mr.x_adj', 'rb')
    # print(len(a))
    # dev_size = 800

    # 训练集和验证集
    with open("data/R8/ind.R8.{}".format('y'), 'rb') as f:  # 看训练集的大小
        y = pkl.load(f)
    # print(len(y))  # 6398
    # input('leny')
    names_train = ['allx_adj', 'allx_embed', 'allx_bert_ids', 'allx_segment_ids', 'ally']  # 训练和验证集
    objects = []
    for i in range(len(names_train)):
        with open("data/R8/ind.R8.{}".format(names_train[i]), 'rb') as f:
            objects.append(pkl.load(f))

    alladj, allembed, allbertids, allsegmentids, ally = tuple(objects)
    train_adj = []
    train_embed = []
    train_bert_ids = []
    train_segment_ids = []
    for i in range(len(ally)):  # 训练和验证集
        a_i = alladj[i].toarray()
        embed_i = np.array(allembed[i])
        train_adj.append(a_i)
        train_embed.append(embed_i)

        bert_ids = np.array(allbertids[i])
        segment_ids = np.array(allsegmentids[i])
        train_bert_ids.append(bert_ids)
        train_segment_ids.append(segment_ids)

    train_y = np.array(ally)
    train_data = []
    dev_data = []
    for i in range(0, len(y)):  # train
        dicti = {'adj': train_adj[i], 'embed': train_embed[i], 'bert_ids': train_bert_ids[i],
                 'segment_ids': train_segment_ids[i], 'y': train_y[i]}
        train_data.append(dicti)

    for i in range(len(y), len(train_bert_ids)):  # eval
        dicti = {'adj': train_adj[i], 'embed': train_embed[i], 'bert_ids': train_bert_ids[i],
                 'segment_ids': train_segment_ids[i], 'y': train_y[i]}
        dev_data.append(dicti)

    #
    names_test = ['tx_adj', 'tx_embed', 'tx_bert_ids', 'tx_segment_ids', 'ty']
    objects = []
    for i in range(len(names_test)):
        with open("data/R8/ind.R8.{}".format(names_test[i]), 'rb') as f:
            objects.append(pkl.load(f))

    tadj, tembed, tbertids, tsegmentids, ty = tuple(objects)
    test_adj = []
    test_embed = []
    test_bert_ids = []
    test_segment_ids = []
    for i in range(len(ty)):
        a_i = tadj[i].toarray()
        embed_i = np.array(tembed[i])
        test_adj.append(a_i)
        test_embed.append(embed_i)

        bert_ids = np.array(tbertids[i])
        segment_ids = np.array(tsegmentids[i])
        test_bert_ids.append(bert_ids)
        test_segment_ids.append(segment_ids)

    test_y = np.array(ty)

    test_data = []
    for i in range(0, len(test_bert_ids)):
        dicti = {'adj': test_adj[i], 'embed': test_embed[i], 'bert_ids': test_bert_ids[i],
                 'segment_ids': test_segment_ids[i], 'y': test_y[i]}
        test_data.append(dicti)

    train_dataset = Data_Dataset(train_data, args)
    dev_dataset = Data_Dataset(dev_data, args)
    test_dataset = Data_Dataset(test_data, args)

    return train_dataset, dev_dataset, test_dataset


class Data_Dataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

        self.convert_features()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        items = d['adj'], d['embed'], torch.tensor(d['bert_ids']), torch.tensor(d['segment_ids']), d['text_len'], d['y']

        # items_tensor = tuple(torch.tensor(t) for t in items)
        items_tensor = tuple(t for t in items)

        # print(items_tensor)
        # input('1')
        return items_tensor

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.data)):
            # self.convert_features_bert(i)
            self.data[i]['text_len'] = len(self.data[i]['bert_ids'])-2



def pad_batch(batch):

    adjs, features, label = zip(*batch)
    label = torch.tensor(label)

    adjs = list(adjs)
    features = list(features)

    # pad adj
    max_length = max([a.shape[0] for a in adjs])
    mask = np.zeros((len(adjs), max_length, 1))

    for i in range(len(adjs)):
        adj_normalized = normalize_adj(adjs[i])  # no self-loop
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adjs[i].shape[0], :] = 1.
        adjs[i] = adj_normalized

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


def pad_batch_bert_gnn(batch):
    adjs, features, input_ids, segment_idx, text_len, tag = zip(*batch)
    tag = torch.tensor(tag)
    text_len = torch.tensor(text_len)


    # gnn
    adjs = list(adjs)
    features = list(features)
    max_length = max([a.shape[0] for a in adjs])
    mask = np.zeros((len(adjs), max_length, 1))

    for i in range(len(adjs)):
        adj_normalized = normalize_adj(adjs[i])
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adjs[i].shape[0], :] = 1.
        adjs[i] = adj_normalized

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


    # bert
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    segment_idx = pad_sequence(segment_idx, batch_first=True, padding_value=0)
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    tag = tag[sorted_idx]
    segment_idx = segment_idx[sorted_idx]


    return adjs, mask, features, input_ids, segment_idx, text_len, tag





def pad_batch_bert(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.
    '''

    input_ids, segment_idx, text_len, tag = zip(*batch)
    text_len = torch.tensor(text_len)
    tag = torch.tensor(tag)

    # Pad sequences.
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    segment_idx = pad_sequence(segment_idx, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    tag = tag[sorted_idx]
    segment_idx = segment_idx[sorted_idx]

    return input_ids, segment_idx, text_len, tag




def pad_batch_train(batch):

    adjs, features, label = zip(*batch)
    label = torch.tensor(label)

    adjs = list(adjs)
    features = list(features)

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

    features = torch.Tensor(features)

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
        # print(adjs[i].shape[0])
        adj_normalized = normalize_adj(adjs[i])  # no self-loop
        pad = max_length - adj_normalized.shape[0]  # padding for each epoch
        # print(adj_normalized.shape[0])
        # print(pad)
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

















