import torch
import torch.nn as nn
from pathlib import Path
import os
import argparse
import numpy as np
from torch.optim import AdamW
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils import class_weight
from collections import Counter
from sklearn.metrics import classification_report
import random
import time
from datetime import datetime
import gc
import pickle
from utils import create_weight_dict, load_data, save_checkpoint, load_checkpoint
from ensemble_loss import alternative_decorrelation_ensemble_error
from run_utils import semevalA_2016_run, wtwt_run
from models import Ensemble


def run(args, model, train_dataloader, validation_dataloader, test_dataloader, label, plot_path):
    alternate = False

    print(' ****** Name of the run is: {}'.format(args.run_number))
    epoch_track_dir = args.main_epoch_dir + args.run_number
    pred_and_true_labels_dir = args.main_pred_dir + args.run_number
    best_model_dir = args.trained_model_dir + args.run_number
    os.mkdir(epoch_track_dir)
    os.mkdir(pred_and_true_labels_dir)
    os.mkdir(best_model_dir)

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

        # pickle.dump([pred, pred_out, real], open('pred-real.pkl', 'wb'))
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Train data batch size')
    parser.add_argument('--val_test_batch_size', type=int, default=8, help='Valid data batch size')
    parser.add_argument('--num_experts', type=int, help='Number of experts (models) in Ensemble')
    parser.add_argument('--task', type=str, help='Task for Stance Detection')
    parser.add_argument('--entity', type=str, help='Entity of financial institution of which you want to test in WTWT dataset')
    parser.add_argument('--learning_rate', type=float, default=2e-5,help='learning rate')
    parser.add_argument('--lambda_', type=int, help='Lambda in decorrelation model error')
    parser.add_argument('--momentum', type=float, default=0.9,help='Momentum in learning rate')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--epochs', type=int, default=30, help='Num of epochs')
    parser.add_argument('--alpha', type=float, help='Reduction factor in Loss function')
    parser.add_argument('--optim_w_decay', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--data_name', type=str, help='Dataset to be used')
    parser.add_argument('--emb_max_len', type=int, help='Size of max length of text embedding')
    parser.add_argument('--num_classes', type=int, help='number of outputs')
    parser.add_argument('--run_num', type=int, help='run number')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # writer = SummaryWriter('./TB_runs/')
    plot_path= f'./plots/{args.model_name}/run_{args.run_num}/'
    model_path= f'./saved_models/{args.model_name}/run_{args.run_num}/'
    tensorboard_path = f'./tensorboard/{args.model_name}/run_{args.run_num}/'
    # writer = SummaryWriter(tensorboard_path)
    if not os.path.exists(path=plot_path) and not os.path.exists(path=model_path) and not os.path.exists(path=tensorboard_path):
        os.mkdir(plot_path) 
        os.mkdir(model_path) 
        os.mkdir(tensorboard_path) 
    else:
        print(f"The file {plot_path} and {model_path} and {tensorboard_path} already exist.")

    ds = {'wtwt': 4, 'semeval_A': 3, 'semeval_A_new_loss': 3, 'semeval-T': 3}

    if args.task == 'semeval_A':
        ds = {'wtwt': 4, 'semeval_A': 3, 'semeval_A_new_loss': 3, 'semeval-T': 3}
        op_towards = {'wtwt': None, 'semeval_A': 3, 'semeval_A_new_loss': 3, 'semeval-T': 3}
        sentiment = {'wtwt': None, 'semeval_A': 3, 'semeval_A_new_loss': 3, 'semeval-T': 3}

        train_df, test_df, train_op_sent_df, test_op_sent_df_, train_size, val_size = load_data(args.task)
        test_op_sent_df_ = test_op_sent_df_[['Opinion towards', 'Sentiment']]
        train_op_sent_df = train_op_sent_df[['Opinion towards', 'Sentiment']]
        
        entire_test_df = pd.concat((test_df, test_op_sent_df_), axis=1)
        entire_train_df = pd.concat((train_df, train_op_sent_df), axis=1)
        # Move the model to the GPU if available
        entire_df = pd.concat((entire_train_df, entire_test_df), axis=0)

        target_map_dict = {'OTHER': 0, 'TARGET': 1, 'NO ONE': 2}
        sent_map_dict = {'NEGATIVE': 0, 'POSITIVE': 1, 'NEITHER': 2}
        entire_df['Opinion towards'] = entire_df['Opinion towards'].apply(lambda x: target_map_dict[x])
        entire_df['Sentiment'] = entire_df['Sentiment'].apply(lambda x: sent_map_dict[x])

        target_vectors = pickle.load(f'{args.main_dir}/codes_semeval/semeval_preprocessed_data/target_embeddings.pkl', 'rb')
        dict_values = pickle.load(open(f'{args.main_dir}/codes_semeval/embeddings/semeval_without_trial/{args.task}_{args.emb_max_len}_processed.pkl', 'rb'))

        # Load
        all_mask = np.array(dict_values['mask'])
        input_ids = np.array(dict_values['input_ids'])
        label = dict_values['label'].to_numpy()

        target_label = np.array(entire_df['Opinion towards'])
        sent_label = np.array(entire_df['Sentiment'])

        train_index_portion = dict_values['train_size']
        val_index_portion = dict_values['val_size']

        random.seed(42)
        all_ids = list(range(len(label)))
        # random.shuffle(all_ids)

        if train_index_portion is None:
            train_size, val_size, test_size = 0.7, 0.1, 0.2
            train_index_portion = int(len(all_ids) * train_size)
            val_index_portion = int(len(all_ids) * (train_size + val_size))
        elif val_index_portion is None:
            val_index_portion = train_index_portion
            train_index_portion = int(val_index_portion * 0.9)
        # indexes
        train_index = all_ids[:train_index_portion]
        val_index = all_ids[train_index_portion: val_index_portion]
        test_index = all_ids[val_index_portion:]

        # create vectorized representation
        input_train = input_ids[train_index]
        input_val = input_ids[val_index]
        input_test = input_ids[test_index]

        # Create attention masks
        mask_train = all_mask[train_index]
        mask_val = all_mask[val_index]
        mask_test = all_mask[test_index]

        print('Subsets saved')
        print("Tokenize the first sentence:")

        train_inputs = torch.Tensor(input_train)
        validation_inputs = torch.Tensor(input_val)
        test_inputs = torch.Tensor(input_test)
        train_labels = torch.Tensor(label[train_index])
        validation_labels = torch.Tensor(label[val_index])
        test_labels = torch.Tensor(label[test_index])
        train_masks = torch.Tensor(mask_train)
        validation_masks = torch.Tensor(mask_val)
        test_masks = torch.Tensor(mask_test)

        # ------------------------------ Targets  ------------------------------
        train_target_labels = torch.Tensor(target_label[train_index])
        validation_target_labels = torch.Tensor(target_label[val_index])
        test_target_labels = torch.Tensor(target_label[test_index])
        print('Tensor created')

        # ------------------------------ Sentiments  ------------------------------
        train_sent_labels = torch.Tensor(sent_label[train_index])
        validation_sent_labels = torch.Tensor(sent_label[val_index])
        test_sent_labels = torch.Tensor(sent_label[test_index])

        train_batch_size = 32
        val_test_batch_size = 8
        # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
        # with an iterator the entire dataset does not need to be loaded into memory

        input_target_train = target_vectors[:train_index_portion]
        input_target_val = target_vectors[train_index_portion:val_index_portion]
        input_target_test = target_vectors[val_index_portion:]

        input_target_train = torch.Tensor(input_target_train)
        input_target_val = torch.Tensor(input_target_val)
        input_target_test = torch.Tensor(input_target_test)

        train_data = TensorDataset(train_inputs, train_masks, input_target_train, train_labels, train_target_labels, train_sent_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=6)

        validation_data = TensorDataset(validation_inputs, validation_masks, input_target_val, validation_labels, validation_target_labels, validation_sent_labels)
        validation_sampler = SequentialSampler(validation_data)
        valid_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.val_test_batch_size, num_workers=6)

        test_data = TensorDataset(test_inputs, test_masks, input_target_test, test_labels, test_target_labels, test_sent_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.val_test_batch_size)

    elif args.task == 'wtwt':
        entity = 'AET_HUM'
        data_dict = pickle.load(open(os.path.join(args.main_dir, f'dataset/Will_they_wont_they/latest_wtwt_entity_embeddings/{args.entity}_as_test_data.pkl', 'rb')))
        random.seed(42)
        all_ids = list(range(len(data_dict['training_data_input_ids'])))  # length as much as the training + validation portion of data

        train_size, val_size = 0.7, 0.3

        train_index_portion = int(len(all_ids) * train_size)
        val_index_portion = int(len(all_ids) * (train_size + val_size))

        train_index = all_ids[:train_index_portion]
        input_train = data_dict['training_data_input_ids'][:train_index_portion]
        mask_train = data_dict['training_data_all_masks'][:train_index_portion]
        train_labels = data_dict['training_labels'][:train_index_portion]

        input_val = data_dict['training_data_input_ids'][train_index_portion:val_index_portion]
        mask_val = data_dict['training_data_all_masks'][train_index_portion:val_index_portion]
        val_labels = data_dict['training_labels'][train_index_portion:val_index_portion]

        input_test = data_dict['test_data_input_ids']
        mask_test = data_dict['test_data_all_masks']
        test_labels = data_dict['test_labels']

        train_inputs = torch.tensor(input_train)
        validation_inputs = torch.tensor(input_val)
        test_inputs = torch.tensor(input_test)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(val_labels)
        test_labels = torch.tensor(test_labels)
        train_masks = torch.tensor(mask_train)
        validation_masks = torch.tensor(mask_val)
        test_masks = torch.tensor(mask_test)

        label = np.concatenate((train_labels, val_labels, test_labels))

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,  num_workers=0)
        # train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.val_test_batch_size, num_workers=0)
        
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.val_test_batch_size)

    if args.model_name == 'ensemble' and args.task == 'semeval_A':
        model = Ensemble(ds_output=ds[args.task], op_output=op_towards[args.task], sent_output=sentiment[args.task]).to(device)
    elif args.model_name == 'ensemble' and args.task == 'wtwt':
        model = Ensemble(ds[args.task])
        
    if args.task == 'semeval_A':
        semevalA_2016_run(args, model, train_dataloader, validation_dataloader, test_dataloader, label, train_index, plot_path, device)
    elif args.task == 'wtwt':
        wtwt_run(args, model, train_dataloader, validation_dataloader, test_dataloader, label, train_index, device)


# Command: python main.py --main_dir C:/Users/jwkor/Documents/Yonsei/Stance Detection --num_experts 3 --task semeval_A --lambda_ 1 --emb_max_len 53 --num_classes 3 --run_num 1
