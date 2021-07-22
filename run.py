import argparse
from model import GNN, Pure_Bert, BERTGNN
import torch.optim as optim
import torch
from dataset import *
import logging
from trainer import train
# from dataloader import TextIGNGraphDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dataset', default='R8', help='Training dataset')
# parser.add_argument('--learning_rate', default=5e-3, type=float, help='Initial learning rate.')
parser.add_argument('--learning_rate', default=5e-5, type=float, help='Initial learning rate.')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, help='Size of batches per epoch.')
# parser.add_argument('--batch_size', default=4096, type=int, help='Size of batches per epoch.')
parser.add_argument('--input_dim', default=300, type=int, help='Dimension of input.')
parser.add_argument('--hidden', default=96, type=int, help='Number of units in hidden layer.')  # 32, 64, 96, 128
parser.add_argument('--steps', default=2, type=int, help='Number of graph layers.')
parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight for L2 loss on embedding matrix.')  # 5e-4
parser.add_argument('--max_degree', default=3, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--early_stopping', default=-1, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--logging_steps', default=20, help='perform evaluate and logging every logging steps.')
parser.add_argument('--num_classes', default=2, help='perform evaluate and logging every logging steps.')
parser.add_argument('--require_improvement', default=100, help='')
parser.add_argument('--model_save_path', default='model_save/' + 'R8_bert_gnn' + '.ckpt', help='')

args = parser.parse_args()

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


logger.info('dataset: %s, learning_rate: %f, epochs: %d, \n batch_size: %d, input_dim: %d, hidden: %d, steps: %d, \n'
            'dropout: %f, weight_decay: %f, early_stopping: %s, logging_steps: %s', args.dataset, args.learning_rate,
            args.epochs, args.batch_size, args.input_dim, args.hidden, args.steps, args.dropout, args.weight_decay,
            args.early_stopping, args.logging_steps)



device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
args.device = device
logger.info('Device is %s', args.device)



train_dataset, dev_dataset, test_dataset = load_datasets(args)
model = BERTGNN(args)



#
pretrained_dict = torch.load('model_save/R8.ckpt')
model_dict = model.state_dict()
# filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)



model.to(args.device)

train(args, train_dataset, model, test_dataset, dev_dataset)





















