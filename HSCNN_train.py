import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from pytorchtools import EarlyStopping


def train(train_pair_iter, val_pair_iter, model, patience):

    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    LR = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses = []
    epochs = 100
    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(epochs):

        for batch_id, batch in enumerate(train_pair_iter):

            text1 = batch.text1
            text2 = batch.text2
            label = batch.label
            label1 = batch.onehot1
            label2 = batch.onehot2
            label = label.unsqueeze(1)
            label = label.clone().detach().requires_grad_(True)

            if torch.cuda.is_available():
                text1 = text1.cuda()
                text2 = text2.cuda()
                label = label.cuda()
                label1 = label1.cuda()
                label2 = label2.cuda()

            optimizer.zero_grad()
            out1, out2, cnn_out1, cnn_out2, out = model.forward_sia(text1, text2, label1, label2,state1='train')

            loss1 = loss_fn(out, label)
            loss2 = loss_fn(cnn_out1, label1)
            loss3 = loss_fn(cnn_out2, label2)
            loss = loss1 + loss2 + loss3

            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        val_loss = eval(val_pair_iter, model)

        train_loss = np.average(train_losses)
        valid_loss = np.average(val_loss)
        avg_train_losses.append(train_loss)
        avg_val_losses.append(valid_loss)
        print('SiaTrain\t[{}/{}]\ttrain_loss:{}\tvalid_loss:{}'.format(epoch, epochs, train_loss, valid_loss))

        train_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        model.load_state_dict(torch.load('./checkpoint/checkpoint.pt'))

    return model, avg_train_losses, avg_val_losses


def eval(valid_pairs_iter, model):
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_val = 0
    val_losses = []
    for batch_id, batch in enumerate(valid_pairs_iter):
        text1 = batch.text1
        text2 = batch.text2
        label = batch.label
        label1 = batch.onehot1
        label2 = batch.onehot2
        label = label.unsqueeze(1)

        if torch.cuda.is_available():
            text1 = text1.cuda()
            text2 = text2.cuda()
            label = label.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()

        out1, out2, cnn_out1, cnn_out2, out = model.forward_sia(text1, text2, label1, label2, state1='train')
        loss = loss_fn(out, label)
        loss_val += loss.item()
        val_losses.append(loss.item())

    return val_losses

