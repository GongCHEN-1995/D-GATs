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

def AUC(outputs,targets):
    global dataset
    assert len(outputs) == len(targets)
    num_targets = len(targets[0])
        
    auc = np.zeros(num_targets)
    targets = np.array(targets).reshape(-1,num_targets)
    outputs = np.array(outputs).reshape(-1,num_targets)

    for i in range(num_targets):
        tmp_targets = targets[:,i]
        tmp_output = outputs[:,i][tmp_targets>=0]
        tmp_targets = tmp_targets[tmp_targets>=0]
        sort_outputs = np.argsort(tmp_output)
        tmp_targets = tmp_targets[sort_outputs]
        arg_pos = np.argwhere(tmp_targets==1)
        num_pos = len(arg_pos)
        num_neg = len(tmp_targets) - num_pos
        if num_pos * num_neg == 0:
            auc[i] = 0
        else:
            auc[i] = (np.sum(arg_pos) - num_pos*(num_pos-1)/2) / num_pos / num_neg
    return auc
        
def Train_NN(dataset, model, train_dl, val_dl, test_dl, best_score, epochs=1, start_lr=1e-6, end_lr=0.1, rate=0.99, step_size=1, store_name=None, metrics='AUC', target_type='classification', mean=None, std=None):     
    loss_MAE = nn.L1Loss()
    loss_MSE = nn.MSELoss()
    if target_type == 'classification':
        if dataset == 'BACE':
            nb_bce = 1
        else:
            nb_bce = train_dl[0][1].size(1)

        all_targets = []
        pos_weight = np.zeros((2,nb_bce))
        for i,j in train_dl:
            all_targets += j.tolist()
        all_targets = np.array(all_targets)

        for i in range(nb_bce):
            pos_weight[0,i] = np.sum(all_targets[:,i]==0)
            pos_weight[1,i] = np.sum(all_targets[:,i]==1)

        pos_weight = pos_weight[0,:] / pos_weight[1,:]
        pos_weight = torch.Tensor(pos_weight).cuda()
        loss_BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif target_type == 'regression':
        if torch.cuda.is_available():
            mean = mean.cuda()
            std = std.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=rate)
    bool_scheduler_step = True
    
    for epoch in range(1,epochs+1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        for p,data in enumerate(train_dl):
            inputs, targets = copy.deepcopy(data)
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                targets = targets.to('cuda')

            # forward + backward + optimize
            outputs = model(inputs)

            if target_type == 'classification':
                if dataset == 'BACE':
                    loss1 = loss_BCE(outputs[:,0],targets[:,0])
                    loss2 = loss_MSE(outputs[:,1],targets[:,1])
                    loss = loss1 + 1.*loss2
                else:
                    mask_void = targets < -0.5
                    loss = loss_BCE(outputs.masked_fill(mask_void,-1e10),targets.masked_fill(mask_void,0))
            elif target_type == 'regression':
                outputs = outputs * std + mean
                if metrics == 'RMSE':
                    loss = loss_MSE(outputs,targets)
                elif metrics == 'MAE':
                    loss = loss_MAE(outputs,targets)
            train_loss += loss.item()
            loss.backward()
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
        if metrics == 'RMSE':
            train_loss = np.sqrt(train_loss)
        torch.cuda.empty_cache()
        val_loss, val_auc = Test_NN(dataset, model, val_dl, metrics, target_type, mean, std)
        test_loss, test_auc = Test_NN(dataset, model, test_dl, metrics, target_type, mean, std)
        torch.cuda.empty_cache()
        if val_auc.max() > 0 and 0 in val_auc:
            val_auc = val_auc[val_auc.nonzero()]
        if test_auc.max() > 0 and 0 in test_auc:
            test_auc = test_auc[test_auc.nonzero()]
        
        val_auc = np.mean(val_auc)
        test_auc = np.mean(test_auc)
        val_loss = np.mean(val_loss)
        test_loss = np.mean(test_loss)
        
        if store_name:
            if target_type == 'regression' and val_loss < best_score[1]:
                best_score = [train_loss, val_loss, test_loss, val_auc, test_auc]
                torch.save(model.to('cpu').state_dict(), './model/'+store_name)
                model = model.to("cuda" if torch.cuda.is_available() else "cpu")
                print('New best model saved!')
            elif target_type == 'classification' and val_auc > best_score[3]:
                best_score = [train_loss, val_loss, test_loss, val_auc, test_auc]
                torch.save(model.to('cpu').state_dict(), './model/'+store_name)
                model = model.to("cuda" if torch.cuda.is_available() else "cpu")
                print('New best model saved!')
        else:
            if target_type == 'regression' and val_loss < best_score[1]:
                best_score = [train_loss, val_loss, test_loss, val_auc, test_auc]
                print('New best model!')
            elif target_type == 'classification' and val_auc > best_score[3]:
                best_score = [train_loss, val_loss, test_loss, val_auc, test_auc]
                print('New best model!')
        elapsed = time.time() - epoch_start_time
        if target_type == 'classification':
            print('| {:2d}/{:2d} epochs | lr {:.1e} | {:2.0f} s | Train/Val/Test loss {:.3e} {:.3e} {:.3e} | AUC {:.5f} {:.5f}'.format(
                epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed, train_loss, val_loss, test_loss, val_auc, test_auc))
        elif target_type == 'regression':
            print('| {:2d}/{:2d} epochs | lr {:.1e} | {:2.0f} s | Train {:.5f} | Val {:.5f} | Test {:.5f}'.format(
                epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed, train_loss, val_loss, test_loss))

    print('The Training is finished: {:.3e} {:.3e} {:.3e} {:.5f} {:.5f}'.format(best_score[0], best_score[1], best_score[2], best_score[3], best_score[4]))
    model.eval()
    return model, best_score

def Test_NN(dataset, model, dl, metrics='AUC', target_type='classification', mean=None, std=None):
    model.eval()
    Loss = 0
    all_targets = []
    all_outputs = []
    num = 0
    
    loss_MSE = nn.MSELoss(reduction = 'sum')
    loss_MAE = nn.L1Loss(reduction = 'sum')
    if target_type == 'classification':
        loss_BCE = nn.BCEWithLogitsLoss(reduction = 'sum')
    elif target_type == 'regression':
        if torch.cuda.is_available():
            mean = mean.cuda()
            std = std.cuda()
            
    with torch.no_grad():
        for p,data in enumerate(dl):
            inputs, targets = copy.deepcopy(data)
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                targets = targets.to('cuda')
    
            # forward + backward + optimize
            outputs = model(inputs)
            if target_type == 'classification':
                if dataset == 'BACE':
                    targets = targets[:,:1]
                    outputs = outputs[:,:1]
                all_targets += targets.tolist()
                all_outputs += torch.sigmoid(outputs).tolist()
                efficient_data = targets >= 0.
                outputs = outputs[efficient_data].reshape(-1,1)
                targets = targets[efficient_data].reshape(-1,1)
                Loss += loss_BCE(outputs,targets).item()
            elif target_type == 'regression':
                outputs = outputs * std + mean
                if metrics == 'RMSE':
                    Loss += loss_MSE(outputs,targets).item()
                elif metrics == 'MAE':
                    Loss += loss_MAE(outputs,targets).item()
            num += outputs.size(0) * outputs.size(1)

    Loss /= num
    if metrics == 'RMSE':
        Loss = np.sqrt(Loss)
        
    if target_type == 'classification':
        auc = AUC(all_outputs, all_targets)
    elif target_type == 'regression':
        auc = np.zeros(1)
    return Loss, auc

def Train_eval(dataset, model, train_dl, val_dl, test_dl, best_score, config_file_path, store_name, metrics, target_type, mean, std):
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
        model, best_score = Train_NN(dataset, model, train_dl, val_dl, test_dl, best_score, epochs[0], start_lr, end_lr, rate, step_size, store_name, metrics, target_type, mean, std)
        print('1st training is finished:' )
        print('Train loss: {:.3e} | Val loss: {:.3e} | Test loss: {:.3e}'.format(best_score[0], best_score[1], best_score[2]))
        if target_type == 'classification':
            print('Val AUC: {:.5f} | Test AUC: {:.5f}'.format(best_score[3], best_score[4]))
        print('')
            
    if epochs[1]:
        for name,parameters in model.named_parameters():
            if 'GAT' in name and 'ReadOut' not in name:
                parameters.requires_grad = False
            else:
                parameters.requires_grad = True
        print('For second training part:')
        model, best_score = Train_NN(dataset, model, train_dl, val_dl, test_dl, best_score, epochs[1], start_lr, end_lr, rate, step_size, store_name, metrics, target_type, mean, std)
        print('2nd training is finished:' )
        print('Train loss: {:.3e} | Val loss: {:.3e} | Test loss: {:.3e}'.format(best_score[0], best_score[1], best_score[2]))
        if target_type == 'classification':
            print('Val AUC: {:.5f} | Test AUC: {:.5f}'.format(best_score[3], best_score[4]))
        print('')
        
    if epochs[2]:
        for name,parameters in model.named_parameters():
            parameters.requires_grad = True
        print('For third training part:')
        model, best_score = Train_NN(dataset, model, train_dl, val_dl, test_dl, best_score, epochs[2], start_lr, end_lr, rate, step_size, store_name, metrics, target_type, mean, std)
        print('3rd training is finished:' )
        print('Train loss: {:.3e} | Val loss: {:.3e} | Test loss: {:.3e}'.format(best_score[0], best_score[1], best_score[2]))
        if target_type == 'classification':
            print('Val AUC: {:.5f} | Test AUC: {:.5f}'.format(best_score[3], best_score[4]))
        print('')
        
    print('Training stage is finished and final results are shown below:')
    print('Train loss: {:.3e} | Val loss: {:.3e} | Test loss: {:.3e}'.format(best_score[0], best_score[1], best_score[2]))
    if target_type == 'classification':
        print('Val AUC: {:.5f} | Test AUC: {:.5f}'.format(best_score[3], best_score[4]))
    return model, best_score