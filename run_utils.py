import os
import torch
import torch.nn as nn
import time
import gc
import pandas as pd
import numpy as np
from datetime import datetime
from torch.optim import AdamW
from collections import Counter
from utils import create_weight_dict, save_checkpoint, load_checkpoint
from ensemble_loss import alternative_decorrelation_ensemble_error, wtwt_loss_update
from sklearn.metrics import classification_report

def semevalA_2016_run(args, model, train_dataloader, validation_dataloader, test_dataloader, label, train_index, plot_path, device):
    alternate = False
  
    print(' ****** Name of the run is: {}'.format(args.run_number))
    pred_and_true_labels_dir = args.main_pred_dir + args.run_number
    os.mkdir(pred_and_true_labels_dir)

    ### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)  # To reproduce BertAdam specific behavior set correct_bias=False
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler

    # Store our loss and accuracy for plotting
    # trange is a tqdm wrapper around the normal python range
    vect_weights = torch.tensor(create_weight_dict(label[train_index]), dtype=torch.float32).to(device)

    loss_fct = nn.CrossEntropyLoss(weight=vect_weights)
    loss_fct.to(device)

    target_loss_fct = nn.CrossEntropyLoss(weight=vect_weights)
    target_loss_fct.to(device)

    sent_loss_fct = nn.CrossEntropyLoss(weight=vect_weights)
    sent_loss_fct.to(device)

    best_loss = np.inf
    best_path = f'./saved_models/{args.task}/mixed_bert_with_concatenated_info_best_{args.run_num}.pt'

    epochs = []                
    epoch_train_loss = []
    epoch_valid_loss = []

    for epoch in range(args.epochs):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        b_train_loss = []
        model.train()
        # Train the data for one epoch
        for i, batch in enumerate(train_dataloader):
            # Add batch to GPU
            torch.cuda.empty_cache()
            gc.collect()
            batch = tuple(t.to(device) for t in batch)
            if i % 100 == 0:
                print(datetime.fromtimestamp(time.time()), i)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_target, b_labels, b_target_labels, b_sent_labels = batch
            len_input = [Counter(value.tolist())[1] for value in b_input_mask]

            ens_sent_outs, ens_OT_out, ens_model_outputs, model_outputs = model(b_input_ids, b_input_mask, len_input, b_target)
            text_error = alternative_decorrelation_ensemble_error(b_labels, model_outputs, num_experts=args.num_experts, num_classes=args.num_classes, alternate=alternate)
            target_error = target_loss_fct(ens_OT_out, b_target_labels.view(-1))
            sent_error = sent_loss_fct(ens_sent_outs, b_sent_labels.view(-1))
            total_error = text_error + target_error + sent_error
            b_train_loss.append(total_error.item())

            total_error.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Forward pass
            # free gpu
            del b_input_ids
            del b_input_mask
            del b_labels
            del b_target_labels
            del b_target
            del batch

        epoch_train_loss.append(np.mean(b_train_loss))
        print(f'{datetime.fromtimestamp(time.time())}\t'+'Epoch [{}/{}]'.format(epoch + 1, args.epochs))

        b_valid_loss = []
        model.eval()
        with torch.no_grad():
            # validation loop
            pred, real, pred_out = [], [], []
            for j, batch in enumerate(validation_dataloader):
                # Add batch to GPU
                if device != 'cpu':
                    batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_target, b_labels, b_target_labels, b_sent_labels = batch
                len_input = [Counter(value.tolist())[1] for value in b_input_mask]
                _,_,outputs,_  = model(b_input_ids, b_input_mask, len_input, b_target)
                pred_out += outputs.tolist()
                pred += outputs.argmax(1).tolist()
                real += b_labels.tolist()

            print(classification_report(real, pred, zero_division=1))
            loss = loss_fct(torch.tensor(pred_out).to(device), torch.tensor(real).to(device))
            b_valid_loss.append(loss)
            print('Loss', loss)
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(best_path, model, 0)
        
        epoch_valid_loss.append(np.mean(b_valid_loss))


    print('---------------------------Best path is: {} --------------------------'.format(best_path))

    load_checkpoint(best_path, model)
    model.eval()
    with torch.no_grad():
        # validation loop
        pred, pred_out, real = [], [], []
        for j, batch in enumerate(test_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_target, b_labels, b_target_labels, b_sent_labels = batch
            len_input = [Counter(value.tolist())[1] for value in b_input_mask]
            _,_,outputs,_ = model(b_input_ids, b_input_mask, len_input, b_target)
            pred += outputs.argmax(1).tolist()
            pred_out += outputs.tolist()
            real += b_labels.tolist()
        print('Tweet Level')
        loss = loss_fct(torch.tensor(pred_out).to(device), torch.tensor(real).to(device))
        print('Loss', loss)
        df_test = pd.DataFrame()
        df_test['pred'] = np.argmax(pred_out, axis=1)
        df_test['real'] = real
        df_test.to_csv(pred_and_true_labels_dir + '/' + f'test_prediction_{args.task}.csv')
        print(args.task)
        print('Test set classification')
        print('-' * 50)
        print(classification_report(real, pred, zero_division=1))

        pred, pred_out, real = [], [], []
        for j, batch in enumerate(validation_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_target, b_labels, b_target_labels, b_sent_labels = batch
            len_input = [Counter(value.tolist())[1] for value in b_input_mask]
            _,_,outputs,_ = model(b_input_ids, b_input_mask, len_input, b_target)
            pred += outputs.argmax(1).tolist()
            pred_out += outputs.tolist()
            real += b_labels.tolist()
        print('Tweet Level')
        loss = loss_fct(torch.tensor(pred_out).to(device), torch.tensor(real).to(device))
        print('Loss', loss)
        df_val = pd.DataFrame()
        df_val['pred'] = np.argmax(pred_out, axis=1)
        df_val['real'] = real
        df_val.to_csv(pred_and_true_labels_dir + '/' + f'val_prediction_{args.task}.csv')
        print(args.task)
        print('Val set classification')
        print('-' * 50)
        print(classification_report(real, pred, zero_division=1))

        pred, pred_out, real = [], [], []
        for j, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_target, b_labels, b_target_labels, b_sent_labels = batch
            len_input = [Counter(value.tolist())[1] for value in b_input_mask]
            _,_,outputs,_ = model(b_input_ids, b_input_mask, len_input, b_target)
            pred += outputs.argmax(1).tolist()
            pred_out += outputs.tolist()
            real += b_labels.tolist()
        print('Tweet Level')
        loss = loss_fct(torch.tensor(pred_out).to(device), torch.tensor(real).to(device))
        print('Loss', loss)
        df_train = pd.DataFrame()
        df_train['pred'] = np.argmax(pred_out, axis=1)
        df_train['real'] = real
        df_train.to_csv(pred_and_true_labels_dir + '/' + f'train_prediction_{args.task}.csv')
        print(args.task)
        print('Train set classification')
        print('-' * 50)
        print(classification_report(real, pred, zero_division=1))

def wtwt_run(args, model, train_dataloader, validation_dataloader, test_dataloader, label, train_index, device):

    print(' ****** Name of the run is: {}'.format(args.run_num))
    pred_and_true_labels_dir = os.path.join(args.main_dir, f'pred_and_true_labels/{args.task}/run_{args.run_num}')
    os.mkdir(pred_and_true_labels_dir)


    lr = 2e-5
    optimizer = AdamW(model.parameters(), lr=lr)  # To reproduce BertAdam specific behavior set correct_bias=False

    # Store our loss and accuracy for plotting
    train_loss_set = []

    vect_weights = create_weight_dict(label[train_index])  # check label here! 
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(vect_weights, dtype=torch.float32))
    loss_fct.to(device)
    best_loss = np.inf
    best_path = os.path.join(args.main_dir, f'saved_models/{args.model_name}/model-full-decorrelated_ensemble-{args.task}-best.pt')

    for epoch in range(args.epochs):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        # Train the data for one epoch
        for i, batch in enumerate(train_dataloader):
            # Add batch to GPU
            torch.cuda.empty_cache()
            gc.collect()
            batch = tuple(t.to(device) for t in batch)
            if i % 100 == 0:
                print(datetime.fromtimestamp(time.time()), i, train_loss_set[-1] if len(train_loss_set) else 0)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            len_input = [Counter(value.tolist())[1] for value in b_input_mask]
            ensemble_output, nets_outputs = model([b_input_ids, b_input_mask])  #  CUDA error: device-side assert triggered when inputs are in cuda
            # print('model_outputs:', model_outputs)
            # print('b_labels:', b_labels)
            loss = wtwt_loss_update(vect_weights, ensemble_output, nets_outputs, b_labels, optimizer)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # free gpu
            del b_input_ids
            del b_input_mask
            del b_labels
            del batch
        
        print(f'{datetime.fromtimestamp(time.time())}\t'+'Epoch [{}/{}]'.format(epoch + 1, args.epochs))

        model.eval()
        with torch.no_grad():
            # validation loop
            pred, real, pred_out = [], [], []
            for j, batch in enumerate(validation_dataloader):
                # Add batch to GPU
                if device != 'cpu':
                    batch = tuple(t.to(device) for t in batch)  # It started to run without issues when this device was turned to cpu!
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                len_input = [Counter(value.tolist())[1] for value in b_input_mask]
                outputs,_ = model([b_input_ids, b_input_mask, len_input])
                pred_out += outputs.tolist()
                pred += outputs.argmax(1).tolist()
                real += b_labels.tolist()
            print(classification_report(real, pred))
            loss = loss_fct(torch.tensor(pred_out).to(device), torch.tensor(real).to(device))
            print('Loss', loss)
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(best_path, model, 0)


    print('---------------------------Best path is: {} --------------------------'.format(best_path))
    load_checkpoint(best_path, model)
    model.eval()
    with torch.no_grad():
        # validation loop
        pred, pred_out, real = [], [], []
        for j, batch in enumerate(test_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            len_input = [Counter(value.tolist())[1] for value in b_input_mask]
            outputs,_ = model([b_input_ids, b_input_mask, len_input])
            pred += outputs.argmax(1).tolist()
            pred_out += outputs.tolist()
            real += b_labels.tolist()
  
        loss = loss_fct(torch.tensor(pred_out).to(device), torch.tensor(real).to(device))
        print('Loss', loss)
        df_test = pd.DataFrame()
        df_test['pred'] = np.argmax(pred_out, axis=1)
        df_test['real'] = real
        df_test.to_csv(pred_and_true_labels_dir + '/' + f'test_prediction_{args.task}.csv')
        print(args.task)
        print('Test set classification')
        print('-' * 50)
        print(classification_report(real, pred))

        pred, pred_out, real = [], [], []
        for j, batch in enumerate(validation_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            len_input = [Counter(value.tolist())[1] for value in b_input_mask]
            outputs,_ = model([b_input_ids, b_input_mask, len_input])
            pred += outputs.argmax(1).tolist()
            pred_out += outputs.tolist()
            real += b_labels.tolist()
 
        loss = loss_fct(torch.tensor(pred_out).to(device), torch.tensor(real).to(device))
        print('Loss', loss)
        df_val = pd.DataFrame()
        df_val['pred'] = np.argmax(pred_out, axis=1)
        df_val['real'] = real
        df_val.to_csv(pred_and_true_labels_dir + '/' + f'val_prediction_{args.task}.csv')
        print(args.task)
        print('Val set classification')
        print('-' * 50)
        print(classification_report(real, pred))

        pred, pred_out, real = [], [], []
        for j, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            len_input = [Counter(value.tolist())[1] for value in b_input_mask]
            outputs,_ = model([b_input_ids, b_input_mask, len_input])
            pred += outputs.argmax(1).tolist()
            pred_out += outputs.tolist()
            real += b_labels.tolist()
       
        loss = loss_fct(torch.tensor(pred_out).to(device), torch.tensor(real).to(device))
        print('Loss', loss)
        df_train = pd.DataFrame()
        df_train['pred'] = np.argmax(pred_out, axis=1)
        df_train['real'] = real
        df_train.to_csv(pred_and_true_labels_dir + '/' + f'train_prediction_{args.task}.csv')
        print(args.task)
        print('Train set classification')
        print('-' * 50)
        print(classification_report(real, pred))


        