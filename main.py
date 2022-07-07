import logging
import torch
import numpy as np
import random
import os
from utils.logger import logger
from data_factory.data_loader import get_loader_segment
import argparse
from train import *
from attten2 import tranModel
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MSL')
parser.add_argument('--model_save_path', type=str, default='./output/')
parser.add_argument('--feature_size', type=int, default=55)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--data_path', default='dataset/MSL')
parser.add_argument('--win_size', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--anormly_ratio', type=float, default=1.00)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--logger_name', type=str, default='test')
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--save_loss', default=True)
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--a', type=int, default=0.1)
parser.add_argument('--b', type=int, default=0.9)
args = parser.parse_args()

if os.path.exists(f'logger/{args.dataset_name}') is True:
    pass
else:
    os.mkdir(f'logger/{args.dataset_name}')

log = logger(f'logger/{args.dataset_name}/{args.logger_name}', logging.DEBUG)
log.debug(f'----------EXP LOGGER NAME :{args.logger_name}----------')
log.debug('----------THIS IS START----------')

log.debug(f'----------SET SEED={args.seed}----------')


def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(seed=args.seed)

log.debug('----------THIS IS PARAMETER ----------')

log.debug(vars(args))

log.debug('----------THIS IS BUILD MODEL ----------')
# model = Encoder(n_head=args.n_head, n_layer=args.n_layers, d_model=args.d_model, dropout=args.dropout,
#                 feature_size=args.feature_size)

model = tranModel(win_size=args.win_size, enc_in=args.feature_size, d_model=args.d_model,
                  n_heads=args.n_head, e_layers=args.n_layers, d_ff=args.d_model, dropout=args.dropout,
                  c_out=args.feature_size)

log.debug(f'----------MODEL READ ----------')

log.debug(f'----------THIS IS READ DATASET:{args.dataset_name}----------')
train_loader = get_loader_segment(args.data_path, batch_size=args.batch_size, win_size=args.win_size,
                                  mode='train',
                                  dataset=args.dataset_name)
vali_loader = get_loader_segment(args.data_path, batch_size=args.batch_size, win_size=args.win_size,
                                 mode='val',
                                 dataset=args.dataset_name)
test_loader = get_loader_segment(args.data_path, batch_size=args.batch_size, win_size=args.win_size,
                                 mode='test',
                                 dataset=args.dataset_name)
thre_loader = get_loader_segment(args.data_path, batch_size=args.batch_size, win_size=args.win_size, step=args.win_size,
                                 mode='thre',
                                 dataset=args.dataset_name)
if args.mode == 'train':
    train(epoch=args.epoch, model=model, model_save_path=args.model_save_path, device=args.device,
          train_loader=train_loader, logger=log, val_loader=vali_loader, lr=args.lr, dataset_name=args.dataset_name,
          logger_name=args.logger_name, save_loss=args.save_loss, n=args.n)
    log.debug('----------THIS IS TRAIN DONE----------')
else:
    test(model=model, model_save_path=args.model_save_path, dataset_name=args.dataset_name, logger=log,
         device=args.device, train_loader=train_loader, thre_loader=thre_loader, anomaly_ratio=args.anormly_ratio,
         logger_name=args.logger_name, a=args.a, b=args.b)
    log.debug('----------THIS IS TEST DONE----------')
