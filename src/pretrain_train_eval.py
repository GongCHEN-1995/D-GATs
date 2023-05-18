import numpy as np
import copy
import time
np.set_printoptions(precision=4)
import json

import torch
import torch.nn as nn

def accuracy_Binary_CLS(outputs, targets, threshold=0.5, return_type='rate'):
    assert return_type in ['rate', 'number']
    
    if isinstance(outputs,torch.Tensor):
        outputs = outputs.to('cpu').numpy()
    if isinstance(targets,torch.Tensor):
        targets = targets.to('cpu').numpy()   
    if isinstance(outputs,list):
        outputs = np.array(outputs)
    if isinstance(targets,list):
        targets = np.array(targets)
    assert isinstance(outputs,np.ndarray)
    assert isinstance(targets,np.ndarray)  

    outputs.reshape(-1)
    targets.reshape(-1)
    
    outputs = outputs > threshold
    accuracy = (outputs-targets) == 0
    accuracy = accuracy.sum()
    if return_type == 'rate':
        accuracy /= outputs.shape[0]
    return accuracy
        
def accuracy_Multi_CLS(outputs,targets,return_type='rate'):
    assert return_type in ['rate', 'number']
    
    if isinstance(outputs,torch.Tensor):
        outputs = outputs.to('cpu').numpy()
    if isinstance(targets,torch.Tensor):
        targets = targets.to('cpu').numpy()   
    if isinstance(outputs,list):
        outputs = np.array(outputs)
    if isinstance(targets,list):
        targets = np.array(targets)
    assert isinstance(outputs,np.ndarray)
    assert isinstance(targets,np.ndarray)  
    
    if targets.ndim != 1:
        targets.reshape(-1)
    assert outputs.shape[1] > 1
    assert outputs.shape[0] == targets.shape[0]
    
    accuracy = outputs.argmax(axis=1) == targets
    accuracy = accuracy.sum()
    
    if return_type == 'rate':
        accuracy /= outputs.shape[0] 
    return accuracy

def Train_NN(model, train_dl, val_dl, test_dl, best_score, epochs=1, start_lr=1e-6, end_lr=0.1, rate=0.99, step_size=1, store_name=None):
    loss_MSE = nn.MSELoss()
    loss_BCE = nn.BCEWithLogitsLoss()
    loss_CEE = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=rate)
    bool_scheduler_step = True

    for epoch in range(1,epochs+1):
        epoch_start_time = time.time()
        model.train()
        train_loss = np.zeros(10)
        for p,data in enumerate(train_dl):
            inputs, targets = copy.deepcopy(data)
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')

            # forward + backward + optimize
            outputs = model(inputs)
            mask = targets['MP'] > 1000
            targets['MP'] = targets['MP'].masked_fill(mask,0)
            outputs['MP'] = outputs['MP'].masked_fill(mask,0)
            loss1 = loss_MSE(outputs['MP'],targets['MP'])
            loss2 = loss_BCE(outputs['ring'],targets['ring'])
            loss3 = loss_BCE(outputs['aromatic'],targets['aromatic'])
            loss4 = loss_CEE(outputs['element'],targets['element'])
            loss5 = loss_CEE(outputs['degree'],targets['degree'])
            loss6 = loss_CEE(outputs['hybridization'],targets['hybridization'])
            loss7 = loss_CEE(outputs['chirality'],targets['chirality'])
            loss8 = loss_CEE(outputs['H'],targets['H'])
            loss9 = loss_MSE(outputs['formal charge'],targets['formal charge'])
            loss10 = loss_MSE(outputs['radical electrons'],targets['radical electrons'])
            train_loss += np.array([loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item(),loss6.item(),loss7.item(),loss8.item(),loss9.item(),loss10.item()])
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_loss, val_acc = Test_NN(model, val_dl)
        test_loss, test_acc = Test_NN(model, test_dl)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.time() - epoch_start_time
        
        print('| {:4d}/{:4d} epoch| lr {:.1e} | {:3.1f} min |'.format(epoch, epochs, optimizer.state_dict()['param_groups'][0]['lr'], elapsed/60))
        if store_name and val_loss.mean() < best_score[1,:10].mean():
            best_score[0,:10] = train_loss
            best_score[1,:10] = val_loss
            best_score[2,:10] = test_loss
            best_score[1,10:] = val_acc
            best_score[2,10:] = test_acc
            torch.save(model.to('cpu').state_dict(), store_name)
            print('New best model', store_name, 'saved!')
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print('Train Loss {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(\
                train_loss[0],train_loss[1],train_loss[2],train_loss[3],train_loss[4],train_loss[5],train_loss[6],train_loss[7],train_loss[8],train_loss[9]))
        print('Val   Loss {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(\
                val_loss[0],val_loss[1],val_loss[2],val_loss[3],val_loss[4],val_loss[5],val_loss[6],val_loss[7],val_loss[8],val_loss[9]))
        print('Test  Loss {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e} {:.2e}'.format(\
                test_loss[0],test_loss[1],test_loss[2],test_loss[3],test_loss[4],test_loss[5],test_loss[6],test_loss[7],test_loss[8],test_loss[9]))
        print('Val  Acc {:.1%} {:.1%} {:.1%} {:.1%} {:.1%} {:.1%} {:.1%}'.format(\
                val_acc[0],val_acc[1],val_acc[2],val_acc[3],val_acc[4],val_acc[5],val_acc[6]))
        print('Test Acc {:.1%} {:.1%} {:.1%} {:.1%} {:.1%} {:.1%} {:.1%} \n'.format(\
                test_acc[0],test_acc[1],test_acc[2],test_acc[3],test_acc[4],test_acc[5],test_acc[6]))
                
    print('The Pre-Training is finished')
    model.eval()
    return model, best_score

def Test_NN(model, dl):
    model.eval()
    loss_MSE = nn.MSELoss()
    loss_BCE = nn.BCEWithLogitsLoss()
    loss_CEE = nn.CrossEntropyLoss()

    loss = np.zeros(10)
    acc = np.zeros(7)
    num = np.zeros(7)
    with torch.no_grad():
        for p,data in enumerate(dl):
            inputs, targets = copy.deepcopy(data)
            if torch.cuda.is_available():
                for name in inputs.keys():
                    inputs[name] = inputs[name].to('cuda')
                for name in targets.keys():
                    targets[name] = targets[name].to('cuda')

            # forward + backward + optimize
            outputs = model(inputs)
            mask = targets['MP'] > 1000
            targets['MP'] = targets['MP'].masked_fill(mask,0)
            outputs['MP'] = outputs['MP'].masked_fill(mask,0)
            loss[0] += loss_MSE(outputs['MP'],targets['MP']).item()
            loss[1] += loss_BCE(outputs['ring'],targets['ring']).item()
            loss[2] += loss_BCE(outputs['aromatic'],targets['aromatic']).item()
            loss[3] += loss_CEE(outputs['element'],targets['element']).item()
            loss[4] += loss_CEE(outputs['degree'],targets['degree']).item()
            loss[5] += loss_CEE(outputs['hybridization'],targets['hybridization']).item()
            loss[6] += loss_CEE(outputs['chirality'],targets['chirality']).item()
            loss[7] += loss_CEE(outputs['H'],targets['H']).item()
            loss[8] += loss_MSE(outputs['formal charge'],targets['formal charge']).item()
            loss[9] += loss_MSE(outputs['radical electrons'],targets['radical electrons']).item()
            acc[0] += accuracy_Binary_CLS(outputs['ring'], targets['ring'], return_type='number')
            acc[1] += accuracy_Binary_CLS(outputs['aromatic'], targets['aromatic'], return_type='number')
            acc[2] += accuracy_Multi_CLS(outputs['element'], targets['element'], return_type='number')
            acc[3] += accuracy_Multi_CLS(outputs['degree'], targets['degree'], return_type='number')
            acc[4] += accuracy_Multi_CLS(outputs['hybridization'], targets['hybridization'], return_type='number')
            acc[5] += accuracy_Multi_CLS(outputs['chirality'], targets['chirality'], return_type='number')
            acc[6] += accuracy_Multi_CLS(outputs['H'], targets['H'], return_type='number')
            num += np.array([len(outputs['ring']), len(outputs['aromatic']), len(outputs['element']), len(outputs['degree']),\
                             len(outputs['hybridization']),len(outputs['chirality']), len(outputs['H'])])
    loss /= len(dl)   
    acc /= num
    return loss, acc

def Train_eval(model, train_dl, val_dl, test_dl, best_score, config_file_path):
    mol_config = json.load(open(config_file_path,'r'))
    start_lr = mol_config["start_lr"]
    end_lr = mol_config["end_lr"]
    rate = mol_config["rate"]
    step_size = mol_config['step_size']
    epochs = mol_config['epochs']
    Save_model_path = mol_config['Save_model_path']

    model, best_score = Train_NN(model, train_dl, val_dl, test_dl, best_score, epochs, start_lr, end_lr, rate, step_size, Save_model_path)
    return model, best_score