import numpy as np
import copy
import time
np.set_printoptions(precision=4)
import functools
import subprocess
from multiprocessing import Pool
import json

import torch
import torch.nn as nn
   
from src.ProbabilityDistribution.Model import PD_NLL_Loss

def Train_NN(dataset, model, train_dl, val_dl, test_dl, best_score, epochs=1, start_lr=1e-6, end_lr=0.1, rate=0.99, step_size=1, store_name=None, target_type='MSE', mean=None, std=None):     
    loss_NLL = PD_NLL_Loss()
    loss_MSE = nn.MSELoss()
    if torch.cuda.is_available():
        mean = mean.cuda()
        std = std.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=rate)
    bool_scheduler_step = True
    
    for epoch in range(1,epochs+1):
        epoch_start_time = time.time()
        model.train()
        train_loss = np.zeros(2)
        for p,data in enumerate(train_dl):
            inputs, targets, PD = copy.deepcopy(data)
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                targets = targets.to('cuda')
                PD = PD.to('cuda')
                
            # forward + backward + optimize
            outputs_MP, outputs_mu, outputs_sigma = model(inputs)
            outputs_MP = outputs_MP * std + mean
            
            loss1 = loss_MSE(outputs_MP, targets)
            if target_type == 'MSE':
                loss2 = loss_MSE(outputs_mu, torch.mean(PD, dim=1)) + loss_MSE(outputs_sigma, torch.std(PD, dim=1))
            elif target_type == 'NLL':
                loss2 = loss_NLL(outputs_mu, outputs_sigma, PD)
            train_loss[0] += loss1.item()
            train_loss[1] += loss2.item()
            
            (loss1+loss2).backward()
            optimizer.step()
#             zero the parameter gradients
            optimizer.zero_grad()

            if rate > 1 and optimizer.state_dict()['param_groups'][0]['lr'] > end_lr:
                bool_scheduler_step = False
            elif rate < 1 and optimizer.state_dict()['param_groups'][0]['lr'] < end_lr:
                bool_scheduler_step = False
            if bool_scheduler_step:
                scheduler.step()
        
        train_loss /= len(train_dl)
        train_loss[0] = np.sqrt(train_loss[0])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_loss = Test_NN(dataset, model, val_dl, target_type, mean, std)
        test_loss = Test_NN(dataset, model, test_dl, target_type, mean, std)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if store_name:
            if val_loss[1] < best_score[1][1]:
                best_score = [train_loss, val_loss, test_loss]
                torch.save(model.to('cpu').state_dict(), './model/'+store_name)
                model = model.to("cuda" if torch.cuda.is_available() else "cpu")
                print('New best model saved!')
        else:
            if val_loss[1] < best_score[1][1]:
                best_score = [train_loss, val_loss, test_loss]
                print('New best model!')
        elapsed = time.time() - epoch_start_time
        print('| {:2d}/{:2d} epochs | lr {:.1e} | {:2.0f} s'.format(epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed))
        print('| Train/Val/Test | Feature Loss: {:.3e},{:.3e},{:.3e} | PD-NLL Loss: {:.3e},{:.3e},{:.3e}\n'.format(train_loss[0], val_loss[0], test_loss[0], train_loss[1], val_loss[1], test_loss[1]))
    model.eval()
    return model, best_score

def Test_NN(dataset, model, dl, target_type='MSE', mean=None, std=None, print_info=False):
    model.eval()
    Loss = np.zeros(2)
    loss_MSE = nn.MSELoss(reduction = 'sum')
    loss_NLL = PD_NLL_Loss(size_average=False)
    num = np.zeros(2)

    if mean is not None and torch.cuda.is_available():
        mean = mean.cuda()
        std = std.cuda()
            
    with torch.no_grad():
        for p,data in enumerate(dl):
            inputs, targets, PD = copy.deepcopy(data)
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                targets = targets.to('cuda')
                PD = PD.to('cuda')
            # forward + backward + optimize
            outputs_MP, outputs_mu, outputs_sigma = model(inputs)
            outputs_MP = outputs_MP * std + mean
            Loss[0] += loss_MSE(outputs_MP, targets).item()
            num[0] += len(targets.view(-1))
            if target_type == 'MSE':
                Loss[1] += loss_MSE(outputs_mu, torch.mean(PD, dim=1)).item() + loss_MSE(outputs_sigma, torch.std(PD, dim=1)).item()
                num[1] += PD.shape[2]
            elif target_type == 'NLL':
                Loss[1] += loss_NLL(outputs_mu, outputs_sigma, PD).item()
                num[1] += PD.shape[1] * PD.shape[2]
            if print_info:
                print('mu prediction: ', outputs_mu)
                print('mu real      : ', torch.mean(PD, dim=1))
                print('sigma prediction: ', outputs_sigma)
                print('sigma real      : ', torch.std(PD, dim=1))
    Loss[0] = np.sqrt(Loss[0] / max(num[0], 1))
    Loss[1] /= max(num[1], 1)
    return Loss

def Train_eval(dataset, model, train_dl, val_dl, test_dl, best_score, config_file_path, store_name, target_type, mean=None, std=None):
    mol_config = json.load(open(config_file_path,'r'))
    start_lr = mol_config["start_lr"]
    end_lr = mol_config["end_lr"]
    rate = mol_config["rate"]
    step_size = mol_config['step_size_' + dataset]
    epochs = mol_config['epochs_' + dataset]

    if epochs[0]:
        for name,parameters in model.named_parameters():
            if 'GAT' in name:
                parameters.requires_grad = False
            else:
                parameters.requires_grad = True
        print('For first training part:')
        model, best_score = Train_NN(dataset, model, train_dl, val_dl, test_dl, best_score, epochs[0], start_lr, end_lr, rate, step_size, store_name, target_type, mean, std)
        print('1st training is finished:' )
        print('Train loss: {:.2e}, {:.2e} | Val loss: {:.2e}, {:.2e} | Test loss: {:.2e}, {:.2e} \n'.format(best_score[0][0], best_score[0][1], best_score[1][0], best_score[1][1], best_score[2][0], best_score[2][1]))
            
    if epochs[1]:
        for name,parameters in model.named_parameters():
            if 'GAT' in name or 'DenseLayer' in name:
                parameters.requires_grad = False
            else:
                parameters.requires_grad = True
        print('For second training part:')
        model, best_score = Train_NN(dataset, model, train_dl, val_dl, test_dl, best_score, epochs[1], start_lr, end_lr, rate, step_size, store_name, target_type, mean, std)
        print('2nd training is finished:' )
        print('Train loss: {:.2e}, {:.2e} | Val loss: {:.2e}, {:.2e} | Test loss: {:.2e}, {:.2e} \n'.format(best_score[0][0], best_score[0][1], best_score[1][0], best_score[1][1], best_score[2][0], best_score[2][1]))
                
    print('Training stage is finished and final results are shown below:')
    print('Train loss: {:.2e}, {:.2e} | Val loss: {:.2e}, {:.2e} | Test loss: {:.2e}, {:.2e} \n'.format(best_score[0][0], best_score[0][1], best_score[1][0], best_score[1][1], best_score[2][0], best_score[2][1]))
    return model, best_score