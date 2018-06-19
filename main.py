import os
import torch
import torchtext.data as data
import argparse
import datetime
import torch.nn.functional as F
import numpy as np
import math

from tqdm import tqdm
from torch.autograd import Variable
from torchtext.datasets import IMDB, SST
from skim_rnn import SkimRNN
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torchtext.vocab import GloVe

parser = argparse.ArgumentParser(description="Skim LSTM")
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.0001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-large_cell_size', type=int, default=100, help='hidden size of large rnn cell [default: 100]')
parser.add_argument('-small_cell_size', type=int, default=5, help='hidden size of small rnn call [default 5]')
parser.add_argument('-num_layers', type=int, default=1, help='number of hidden layer [default 1]')
parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-hidden_layer', type=int, default=200,
                    help='dimension of hidden layer in the fully connected network [default: 200]')
parser.add_argument('-gamma', type=float, default=0.01, help='gamma regularization parameter [default: 0.01]')
parser.add_argument('-tau', type=float, default=0.5, help='gamma regularization parameter [default: 0.5]')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, 0 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

def load_data(text_field, label_field, **kwargs):
    train_data, test_data, _ = SST.splits(text_field, label_field,
                                          filter_pred=lambda ex: ex.label != 'neutral')
    text_field.build_vocab(train_data, vectors=GloVe()),
    label_field.build_vocab(train_data, test_data)
    train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data),
        batch_sizes=(args.batch_size, args.batch_size),
        shuffle=args.shuffle,
        **kwargs
    )
    return train_iter, test_iter


print("\nLoading data...")
text_field = data.Field(batch_first=True, lower=True, tokenize='spacy')
label_field = data.Field(sequential=False, unk_token=None)
train_iter, test_iter = load_data(text_field, label_field, device=-1, repeat=False)

args.vocab_size = len(text_field.vocab)
args.n_class = len(label_field.vocab)
args.word_dict = text_field.vocab

args.cuda = (not args.no_cuda) and torch.cuda.is_available();
del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
if args.snapshot is None:
    skim_rnn_classifier = SkimRNN(args)
else:
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        skim_rnn_classifier = torch.load(args.snapshot)
    except:
        print("Sorry, This snapshot doesn't exist.");
        exit()

if args.cuda:
    skim_rnn_classifier = skim_rnn_classifier.cuda()

optim = torch.optim.Adam(filter(lambda x: x.requires_grad, skim_rnn_classifier.parameters()), lr=args.lr)

global_step = 0
EPSILON = 1e-15
r = 1e-4

for epoch in range(args.epochs):
    step = 0

    scores = np.array([])
    target = np.array([])

    scores_ = np.array([])
    target_ = np.array([])

    skim_rnn_classifier.train()

    for batch in tqdm(train_iter):
        sent, label = batch.text, batch.label

        if torch.cuda.is_available():
            sent, label = sent.cuda(), label.cuda()
        optim.zero_grad()

        tau = max(0.5, math.exp(-r * global_step))
        logits, h_stack, p_stack = skim_rnn_classifier(sent, tau)
        loss = F.cross_entropy(logits, label) # + args.gamma * torch.mean(-torch.log(p_stack[:, :, 1] ** 2 + EPSILON))
        loss.backward()
        optim.step()

        step += 1
        global_step += 1
        predict = torch.max(logits, 1)[1].squeeze()

        scores = np.append(scores, predict.cpu().data.numpy())
        target = np.append(target, label.cpu().data.numpy())

    acc, pre, rec = accuracy_score(target, scores), precision_score(target, scores), recall_score(target, scores)
    print('Epoch [%d/%d] | Step [%d/%d] | Loss: %.4f | Acc: %.4f | Precision: %.4f | Recall: %.4f'
          % (epoch + 1, args.epochs, step + 1, len(train_iter), loss.data.item(), acc, pre, rec))
    scores = np.array([])
    target = np.array([])

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    save_prefix = os.path.join(args.save_dir, 'snapshot')
    save_path = '{}_steps_{}.pt'.format(save_prefix, step)
    torch.save(skim_rnn_classifier, save_path)

    scores_ = np.array([])
    target_ = np.array([])

    skim_rnn_classifier.eval()

    for batch in test_iter:
        sent, label = batch.text, batch.label
        sent, label = sent.cuda(), label.cuda()
        logits, h_stack, Q_stack = skim_rnn_classifier(sent)
        predict = torch.max(logits, 1)[1].squeeze()
        scores_ = np.append(scores_, predict.cpu().data.numpy())
        target_ = np.append(target_, label.cpu().data.numpy())

    acc, pre, rec = accuracy_score(target_, scores_), precision_score(target_, scores_), recall_score(target_, scores_)
    print('Epoch [%d/%d] | Loss: %.4f | Acc: %.4f | Precision: %.4f | Recall: %.4f'
          % (epoch + 1, args.epochs, loss.data.item(), acc, pre, rec))