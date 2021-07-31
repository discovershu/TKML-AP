# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:17:39 2021
@author: Lipeng Ke
"""

from tqdm import tqdm
import torch
import gc
import os
from utils import get_ap_score, generate_target_zeros
import numpy as np
import torch
import numpy as np
from evaluate_metrics import hamming_loss, FR, TP_index, nontargeted_TP_index
from models.projection import clip_eta
from torch.autograd import Variable
from torch import nn

def pairwise_and(a, b):
    column = torch.unsqueeze(a, 2)
    row = torch.unsqueeze(b, 1)
    return torch.logical_and(column, row)

# compute pairwise differences between elements of the tensors a and b
def pairwise_sub(a, b):
    column = torch.unsqueeze(a, 2)
    row = torch.unsqueeze(b, 1)
    return column - row


def loss_fun(y, y_pre, device):
    y_i = torch.eq(y, torch.ones(y.shape).to(device))
    y_not_i = torch.eq(y, torch.zeros(y.shape).to(device))

    # get indices to check
    truth_matrix = pairwise_and(y_i, y_not_i).float()

    # calculate all exp'd differences
    # through and with truth_matrix, we can get all c_i - c_k(appear in the paper)
    sub_matrix = pairwise_sub(y_pre, y_pre)
    exp_matrix = torch.exp(torch.neg(sub_matrix))

    # check which differences to consider and sum them

    sparse_matrix = torch.mul(exp_matrix, truth_matrix)
    sums = torch.sum(sparse_matrix, (1, 2))

    # get normalizing terms and apply them
    y_i_sizes = torch.sum(y_i.float(), (1))
    y_i_bar_sizes = torch.sum(y_not_i.float(), (1))
    normalizers = torch.mul(y_i_sizes , y_i_bar_sizes)

    # negalete the column y that do not have instances
    # loss = torch.mean(torch.div(sums, normalizers))
    loss = torch.mean(torch.div(sums[normalizers!=0],normalizers[normalizers!=0]))

    return loss

def train_model(model, device, optimizer, scheduler, train_loader, valid_loader, save_dir, model_num, epochs, log_file, loss_balance=0.5):
    """
    Train a deep neural network model

    Args:
        model: pytorch model object
        device: cuda or cpu
        optimizer: pytorch optimizer object
        scheduler: learning rate scheduler object that wraps the optimizer
        train_dataloader: training  images dataloader
        valid_dataloader: validation images dataloader
        save_dir: Location to save model weights, plots and log_file
        epochs: number of training epochs
        log_file: text file instance to record training and validation history

    Returns:
        Training history and Validation history (loss and average precision)
    """

    tr_loss, tr_map = [], []
    val_loss, val_map = [], []
    best_val_map = 0.0
    sigmoid = nn.Sigmoid()

    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch+1))
        log_file.write("Epoch {} >>".format(epoch+1))
        scheduler.step()

        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_ap = 0.0

            criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
            m = torch.nn.Sigmoid()

            if phase == 'train':
                model.train(True)  # Set model to training mode

                for data, target in tqdm(train_loader):
                    #print(data)
                    target = target.float()
                    if len(target.shape):
                        target = target.max(dim=1)[0]
                    else:
                        pass
                    data, target = data.to(device), target.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    output = model(data)
                    try:
                        output = output.logits
                    except:
                        pass
                    # output = sigmoid(output)
                    loss = criterion(output, target)

                    # loss = loss_balance * loss_fun(target, output, device)
                    # loss = loss_fun(torch.transpose(target, 0, 1), torch.transpose(output, 0, 1), device)
                    # loss = loss_balance * loss_fun(target, output, device) + loss_fun(torch.transpose(target, 0, 1), torch.transpose(output, 0, 1), device)


                    # Get metrics here
                    running_loss += loss # sum up batch loss

                    running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(sigmoid(output)).detach().numpy())

                    # Backpropagate the system the determine the gradients
                    loss.backward()

                    # Update the paramteres of the model
                    optimizer.step()

                    # clear variables
                    #del data, target, output, loss
                    #gc.collect()
                    #torch.cuda.empty_cache()

                    # print("loss = ", running_loss)

                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss.item()/num_samples
                tr_map_ = running_ap/num_samples

                print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
                    tr_loss_, tr_map_))

                log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
                    tr_loss_, tr_map_))

                # Append the values to global arrays
                tr_loss.append(tr_loss_), tr_map.append(tr_map_)


            else:
                model.train(False)  # Set model to evaluate mode

                # torch.no_grad is for memory savings
                with torch.no_grad():
                    for data, target in tqdm(valid_loader):
                        target = target.float()
                        if len(target.shape) ==3:
                            target = target.max(dim=1)[0]
                        else:
                            pass
                        data, target = data.to(device), target.to(device)
                        output = model(data)

                        loss = criterion(output, target)

                        running_loss += loss # sum up batch loss
                        running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(sigmoid(output)).detach().numpy())

                        #del data, target, output
                        #gc.collect()
                        #torch.cuda.empty_cache()

                    num_samples = float(len(valid_loader.dataset))
                    val_loss_ = running_loss.item()/num_samples
                    val_map_ = running_ap/num_samples

                    # Append the values to global arrays
                    val_loss.append(val_loss_), val_map.append(val_map_)

                    print('val_loss: {:.4f}, val_avg_precision:{:.3f}'.format(
                    val_loss_, val_map_))

                    log_file.write('val_loss: {:.4f}, val_avg_precision:{:.3f}\n'.format(
                    val_loss_, val_map_))

                    # Save model using val_acc
                    if val_map_ >= best_val_map:
                        best_val_map = val_map_
                        log_file.write("saving best weights...\n")
                        torch.save(model.state_dict(), os.path.join(save_dir,"model-{}.pth".format(model_num)))

    return ([tr_loss, tr_map], [val_loss, val_map])



def test(model, device, test_loader, returnAllScores=False, num_classes=20):
    """
    Evaluate a deep neural network model

    Args:
        model: pytorch model object
        device: cuda or cpu
        test_dataloader: test images dataloader
        returnAllScores: If true addtionally return all confidence scores and ground truth

    Returns:
        test loss and average precision. If returnAllScores = True, check Args
    """
    model.train(False)

    running_loss = 0
    running_ap = 0

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    sigmoid = torch.nn.Sigmoid()

    if returnAllScores == True:
        all_scores = np.empty((0, num_classes), float)
        ground_scores = np.empty((0, num_classes), float)

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            #print(data.size(), target.size())
            target = target.float()
            if len(target.shape) == 3:
                target = target.max(dim=1)[0]
            else:
                pass
            data, target = data.to(device), target.to(device)
            # bs, ncrops, c, h, w = data.size()

            # output = model(data.view(-1, c, h, w))
            output = model(data)
            # output = m(output)
            # output = output.view(bs, ncrops, -1).mean(1)

            loss = criterion(output, target)

            running_loss += loss # sum up batch loss
            # running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy())
            running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(),
                                       torch.Tensor.cpu(sigmoid(output)).detach().numpy())

            if returnAllScores == True:
                all_scores = np.append(all_scores, torch.Tensor.cpu(sigmoid(output)).detach().numpy() , axis=0)
                ground_scores = np.append(ground_scores, torch.Tensor.cpu(target).detach().numpy() , axis=0)

            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

    num_samples = float(len(test_loader.dataset))
    avg_test_loss = running_loss.item()/num_samples
    test_map = running_ap/num_samples

    print('test_loss: {:.4f}, test_avg_precision:{:.3f}'.format(
                    avg_test_loss, test_map))


    if returnAllScores == False:
        return avg_test_loss, running_ap

    return avg_test_loss, running_ap, all_scores, ground_scores
