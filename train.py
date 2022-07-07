import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from test_score import score
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef, f1_score
from sklearn.metrics import classification_report


class EarlyStopping:
    def __init__(self, dataset_name, logger_name, logger, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.logger = logger
        self.logger_name = logger_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.logger.debug(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(),
                   os.path.join(path, str(self.logger_name) + str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, lr_, logger):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.debug('Updating learning rate to {}'.format(lr))


def model_train(model, optimizer, loss, device, train_loader, logger, n):
    rec1 = []
    rec2 = []

    model.train()
    model.to(device)
    iter_start = time.time()
    for i, (input_data, labels) in enumerate(train_loader):
        iter_one_start = time.time()
        optimizer.zero_grad()
        input_data = input_data.float().to(device)
        dec_out1, dec_out2 = model(input_data)
        rec_loss1 = loss(input_data, dec_out1)
        rec_loss2 = loss(input_data, dec_out2)
        ######################################################
        loss1 = (1 / n) * rec_loss1 + (1 - 1 / n) * rec_loss2
        loss2 = (1 / n) * rec_loss1 - (1 - 1 / n) * rec_loss2
        rec1.append(loss1.item())
        rec2.append(loss2.item())
        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer.step()
        ######################################################
        iter_one_end = time.time() - iter_one_start
        if i % 5 == 0:
            iter_end = time.time() - iter_start
            left_time = iter_one_end * (len(train_loader) - i)
            logger.debug(
                f'iter : {i + 1} |rec1: {loss1.item():.4f} | rec2: {loss2.item():.4f}   '
                f'|cost_time: {int(iter_end // 3600):d}时{int((iter_end % 3600) // 60):d}分{(iter_end % 3600) % 60:.2f}秒'
                f'|left time:{int(left_time // 3600):d}时{int((left_time % 3600) // 60):d}分{(left_time % 3600) % 60:.2f}秒')
    rec1_res = torch.tensor(rec1).mean()
    rec2_res = torch.tensor(rec2).mean()

    return rec1_res, rec2_res


def model_evaluate(model, device, loss, val_loader,n):
    model.eval()
    total_rec1 = []
    total_rec2 = []
    for i, (input_data, _) in enumerate(tqdm(val_loader)):
        input_data = input_data.float().to(device)
        dec_out1, dec_out2 ,dec_out3= model(input_data)
        rec1 = loss(input_data, dec_out1)
        rec2 = loss(input_data, dec_out2)
        loss1 = (1 / n) * rec1 + (1 - 1 / n) * rec2
        loss2 = (1 / n) * rec1 - (1 - 1 / n) * rec2
        total_rec1.append(loss1.item())
        total_rec2.append(loss2.item())
    total_rec1 = torch.tensor(total_rec1).mean()
    total_rec2 = torch.tensor(total_rec2).mean()

    return total_rec1, total_rec2


def loss_save(path, train_rec=None, val_rec=None, train_rec2=None, val_rec2=None):
    if train_rec is not None:
        train_rec = np.array(train_rec).reshape(-1)
        np.save(path + 'train_rec', train_rec)
    if train_rec2 is not None:
        train_rec2 = np.array(train_rec2).reshape(-1)
        np.save(path + 'train_rec', train_rec2)

    if val_rec is not None:
        val_rec = np.array(val_rec).reshape(-1)
        np.save(path + 'val_rec', val_rec)
    if val_rec2 is not None:
        val_rec2 = np.array(val_rec2).reshape(-1)
        np.save(path + 'val_rec', val_rec2)


def train(epoch, model, model_save_path, device, train_loader, logger, val_loader, lr, dataset_name, logger_name,
          save_loss, n):
    logger.debug('-----------THIS IS TRAIN START----------')
    path = model_save_path + f'{str(logger_name)}_{dataset_name}/'
    if not os.path.exists(path):
        os.makedirs(path)
    logger.debug(f'model_save_path:{path}')
    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=dataset_name, logger=logger,
                                   logger_name=logger_name)
    total_rec1 = []
    total_rec2 = []
    val_total_rec1 = []
    val_total_rec2 = []
    for i in range(epoch):
        logger.debug(f'====================Epoch:{i + 1}====================')
        train_start = time.time()
        rec1, rec2 = model_train(model, optimizer, loss, device, train_loader, logger, n)
        epoch_end = time.time() - train_start
        logger.debug(
            f'Epoch : {i + 1} | rec1 : {rec1:.4f} |rec1 : {rec1:.4f} '
            f'| cost_time : {int(epoch_end // 3600):d}时{int((epoch % 3600) // 60):d}分{(epoch_end % 3600) % 60:.2f}秒')
        total_rec1.append(rec1.item())
        total_rec2.append(rec2.item())
        val_rec1, val_rec2 = model_evaluate(model, device, loss, val_loader,n)

        val_total_rec1.append(val_rec1.item())
        val_total_rec2.append(val_rec2.item())
        train_one_end = time.time() - train_start
        train_left_time = train_one_end * (epoch - i)
        logger.debug('=' * 20 + 'THIS IS VAL' + '=' * 20)
        logger.debug(
            f'Epoch : {i + 1}  rec1 : {val_rec1} |  rec2 : {val_rec2} '
            f'train left time:{int(train_left_time // 3600):d}时{int((train_left_time % 3600) // 60):d}分{(train_left_time % 3600) % 60:.2f}秒')
        early_stopping(val_rec1, model, path)
        if early_stopping.early_stop:
            logger.debug("Early stopping")
            break
        # adjust_learning_rate(optimizer, i + 1, lr, logger)
    if save_loss is True:
        logger.debug('=' * 20 + 'THIS IS SAVE LOSS' + '=' * 20)
        loss_save(path, total_rec1, val_total_rec1, total_rec2, val_total_rec2)


def test(model, model_save_path, dataset_name, logger, device, train_loader, thre_loader, anomaly_ratio, logger_name,a,b):
    path = model_save_path + f'{str(logger_name)}_{dataset_name}/'
    train_score_list = []
    test_labels = []
    model.load_state_dict(torch.load(
        os.path.join(path,
                     str(logger_name) + str(dataset_name) + '_checkpoint.pth')))
    model.eval()
    logger.debug('----------THIS IS TEST START----------')
    # 1 find threshold
    for i, (input_data, labels) in enumerate(tqdm(train_loader)):
        input_data = input_data.float().to(device)
        model = model.to(device)
        dec_out1,dec_out2 = model(input_data)
        ####################################################
        final_score = score(input_data, dec_out1,dec_out2,a,b)
        ####################################################
        train_score_list.append(final_score.detach().cpu().numpy())
    train_score = np.concatenate(train_score_list, axis=0).reshape(-1)
    train_score = np.array(train_score)
    thresh = np.percentile(train_score, 100 - anomaly_ratio)
    logger.debug("###################################################")
    logger.debug(f"Threshold :{thresh}")
    logger.debug("###################################################")

    # 2 test
    test_score_list = []
    for i, (input_data, labels) in enumerate(tqdm(thre_loader)):
        input_data = input_data.float().to(device)
        dec_out1,dec_out2 = model(input_data)
        al_score = score(input_data, dec_out1,dec_out2,a,b)
        test_labels.append(labels.detach().cpu().numpy())
        test_score_list.append(np.array(al_score.detach().cpu().numpy()))
    test_score = np.concatenate(test_score_list, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_score = np.array(test_score)
    test_labels = np.array(test_labels)

    # test_score = train_score_list[0].reshape(-1)
    pred = (test_score > thresh).astype(int)

    gt = test_labels.astype(int)

    logger.debug(f"pred:{pred.shape}")
    logger.debug(f"gt:{gt.shape}")

    # detection adjustment
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    logger.debug(f'pred:  {pred.shape}')
    logger.debug(f'gt:  {gt.shape}')

    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    f_score = f1_score(gt, pred)
    MCC = matthews_corrcoef(gt, pred)
    conf = confusion_matrix(gt, pred)

    logger.debug(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ,MCC : {:0.4F} ".format(
            accuracy, precision,
            recall, f_score, MCC))
    logger.debug(f"\n {conf}")
    logger.debug(f"\n {classification_report(gt, pred)}")
