import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

def create_weight_dict(labels):
    unique_labels = np.unique(labels)
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=unique_labels,
                                                      y=labels)
    return class_weights


def save_plots(epoch_train_loss, epoch_valid_loss, epochs, args):
    plt.figure(figsize=(14, 5))

  # Accuracy plot
    plt.subplot(1, 2, 1)
    train_loss_plot, = plt.plot(epochs, epoch_train_loss, 'r')
    val_loss_plot, = plt.plot(epochs, epoch_valid_loss, 'b')
    plt.title('Training and Validation Loss')
    plt.legend([train_loss_plot, val_loss_plot], ['Training Loss', 'Validation Loss'])
    plt.savefig(f'./plots/{args.model_name}/run_{args.run_num}/loss_plots.jpg')


def save_checkpoint(save_path, model, loss, val_used=None):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'val_loss': loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def load_data(args, name, text_type=None, remove_nan=None):

    # if name == 'semeval':
    topics = ['AT', 'CC', 'FM', 'HC', 'LA']
    whole_train_df = pd.DataFrame(columns = ['ID', 'Target', 'tweet', 'Stance'])
    whole_test_df = pd.DataFrame(columns = ['ID', 'Target', 'tweet', 'Stance'])

    whole_train_op_sent_df = pd.read_csv(f'{args.main_dir}/codes_semeval/semeval_data/trainingdata-all-annotations.txt', sep='\t', encoding='latin-1', index_col=0).reset_index(drop=True)
    whole_train_op_sent_df = whole_train_op_sent_df[['Target', 'Tweet', 'Opinion towards', 'Sentiment']]

    # pd.DataFrame(columns = ['ID', 'Target', 'Opinion towards', 'Sentiment'])
    whole_test_op_sent_df = pd.read_csv(f'{args.main_dir}/codes_semeval/semeval_data/testdata-taskA-all-annotations.txt', sep='\t', encoding='latin-1', index_col=0).reset_index(drop=True)
    whole_test_op_sent_df = whole_test_op_sent_df[['Target', 'Tweet', 'Opinion towards', 'Sentiment']]
    # pd.DataFrame(columns = ['ID', 'Target', 'Opinion towards', 'Sentiment'])

    for topic in topics:
        test_df = pd.read_csv(f'{args.main_dir}/codes_semeval/semeval_preprocessed_data/{topic}/test_clean.txt', sep='\t', encoding='latin-1', index_col=0)
        train_df = pd.read_csv(f'{args.main_dir}/codes_semeval/semeval_preprocessed_data/{topic}/train_clean.txt', sep='\t', encoding='latin-1', index_col=0)
        whole_test_df = whole_test_df.append(test_df).reset_index(drop=True)
        whole_train_df = whole_train_df.append(train_df).reset_index(drop=True)

    whole_train_df = whole_train_df[['tweet', 'Stance', 'Target']]
    whole_test_df = whole_test_df[['tweet', 'Stance', 'Target']]
    map_dict = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}
    whole_train_df['label'] = whole_train_df['Stance'].apply(lambda x: map_dict[x])
    whole_train_df['text'] = whole_train_df['tweet']
    whole_test_df['label'] = whole_test_df['Stance'].apply(lambda x: map_dict[x])
    whole_test_df['text'] = whole_test_df['tweet']
    whole_train_df = whole_train_df.drop(labels=['tweet', 'Stance'], axis=1)
    whole_test_df = whole_test_df.drop(labels=['tweet', 'Stance'], axis=1)
    # df = pd.concat((whole_train_df, whole_test_df)).reset_index(drop=True)
    return whole_train_df, whole_test_df, whole_train_op_sent_df, whole_test_op_sent_df, len(whole_train_df), None