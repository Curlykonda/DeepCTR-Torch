import pandas as pd
import torch
import random
import pickle
import argparse
import string
import itertools
import numpy as np
from tqdm import tqdm

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import *
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def generate_neg_samples(data_point, item_ids, n_ratio, neg_label):
    neg_samples = pd.DataFrame(columns=data_point.columns)
    added_items = []

    neg_sample = data_point.copy()  # pd.Series
    neg_sample['label'] = neg_label

    for i in range(n_ratio):

        new_item = random.choice(tuple(item_ids))

        while new_item in added_items or new_item in data_point[('history')]:
            new_item = random.choice(tuple(item_ids))

        # modify history
        hist = neg_sample['history'].copy()
        hist[-1] = new_item
        neg_sample['history'] = hist

        neg_samples = neg_samples.append(neg_sample, ignore_index=True)
        added_items.append(new_item)

    return neg_samples


def generate_dummy_ids(id_len, n_id, letters=None):
    ids = []
    key2index = {}

    if letters == None:
        allowed_letters = string.ascii_letters + string.digits
    else:
        allowed_letters = letters

    new_id = ''.join(random.choice(allowed_letters) for _ in range(id_len))
    for i in range(n_id):
        j = 0
        while new_id in ids:
            new_id = ''.join(random.choice(allowed_letters) for j in range(id_len))
            j += 1
            if j > 10 * n_id:
                raise TimeoutError("Allowed letter space is too small to create sufficient unique IDs")

        ids.append(new_id)
        key2index[new_id] = len(key2index) + 1

    return ids, key2index


def generate_dummy_history(hist_len, item_ids, n_users, pad_prob=0.3, pad=0):
    hist = []

    for u in range(n_users):
        # fill history with random items
        user_hist = [random.choice(tuple(item_ids)) for i in range(hist_len)]
        # with pad probability erase last N items randomly (not all users have to a full-length reading history)
        if random.random() < pad_prob:
            idx = random.randint(1, hist_len - 2)
            user_hist[-idx:] = [pad] * idx

        hist.append(user_hist)

    return hist


def create_mappings_key2idx(ids):
    item2idx = {key: i + 1 for i, key in enumerate(ids)}  # starting at 1 because 0 is for padding
    idx2item = {val: key for (key, val) in item2idx.items()}

    return item2idx, idx2item

def map_key2idx(key2idx, values):
    return [int(key2idx[val]) for val in values]


def main(config):

    # read / create data
    target = ['label']
    """
    format of data point
    columns = u_id, label, hist [i_id0, i_id1, .. , i_idN]
    """
    load_existing_df = True

    if load_existing_df:
        data = pd.read_pickle(config.load_pkl_data) # dataframe containing neg & pos samples

        with open(config.data_dir + "item_ids_100k.pkl", 'rb') as fin:
            item_ids = set(pickle.load(fin))

        item2idx, idx2item = create_mappings_key2idx(item_ids)
        user_ids = list(data['user_id'].unique())

    else:
        if config.use_dummy_data:
            n_items = int(1e4)
            n_users = int(1e3)
            hist_max_len = config.hist_max_len

            item_ids, item2idx = generate_dummy_ids(id_len=6, n_id=n_items)
            user_ids, user2idx = generate_dummy_ids(id_len=6, n_id=n_users, letters=string.ascii_letters)
            item_ids = set(item_ids)

            print("Generating dummy history..")
            hist = generate_dummy_history(hist_len=hist_max_len, item_ids=item_ids, n_users=n_users, pad_prob=0)

        else:
            # read data
            with open(config.data_dir + "item_ids_100k.pkl", 'rb') as fin:
                item_ids = set(pickle.load(fin))
            with open(config.data_dir + "users_on_item_subset.pkl", 'rb') as fin:
                users = pickle.load(fin)

            user_ids = list(users.keys())
            hist = []

            # create mappings
            item2idx, idx2item = create_mappings_key2idx(item_ids)

            # extract only article_ids from user history
            max_hist_len_in_data = 0

            for u_id in user_ids:
                hist_ids = [i_id for paper, i_id, _ in users[u_id]['articles_read']]
                hist.append(hist_ids)
                if max_hist_len_in_data < len(hist_ids):
                    max_hist_len_in_data = len(hist_ids)

        labels = [1] * len(user_ids)
        idx2item = {val: key for (key, val) in item2idx.items()}

        columns = ["user_id"]
        columns += ["label"]
        columns += ["history"]

        last_column = columns[-1]

        # data = pd.DataFrame([list(itertools.chain(*i)) for i in zip(zip(user_ids, labels), hist)], columns=columns)
        # col2 = ["u_id", "label", "history"]
        data = pd.DataFrame(zip(user_ids, labels, hist), columns=columns)

        # add negative samples
        user_ids = data['user_id']
        if config.neg_sampling_ratio > 0:
            print("Start generating negative samples for {} users..".format(len(user_ids)))
            for u in tqdm(user_ids):
                u_data_p = data.loc[data['user_id'] == u]
                neg_samples = generate_neg_samples(u_data_p, item_ids, n_ratio=config.neg_sampling_ratio, neg_label=0)
                data = data.append(neg_samples, ignore_index=True)

        #save data
        print(">> saving data with Pickle protocol {}".format(pickle.HIGHEST_PROTOCOL))
        data.to_pickle(config.load_pkl_data, protocol=pickle.HIGHEST_PROTOCOL)

    # encode features
    print("Encode features..")
    lbe = LabelEncoder()
    data['user_id'] = lbe.fit_transform(data['user_id'])
    data = data.astype({'user_id': 'int32', 'label': 'int32'})

    # replace item IDs in history with corresponding index from "item vocabulary"
    history_ids = list(map(lambda x: map_key2idx(item2idx, x), data['history'].values))
    #pad & truncate item history to
    history_ids = pad_sequences(history_ids, maxlen=config.hist_max_len, dtype='int32',
                                padding='post', truncating='pre', value=int(0)) #Note: padding token is '0'
    # transform to ndarray
    data['history'] = history_ids
    #data = pd.DataFrame(zip(data['user_id'], data['label'], history_ids), columns=data.columns)

    # Problem description: want to assign 'history_ids' to the dataframe while preserving the correct datatype 'int'.
    # But currently, when setting / replacing the values of history in the df, the datatype also changes;
    # it becomes an sequence of type 'object' not 'int' ...

    # Need to fulfill this condition for subsequent input methods to work properly:
    # assert data['history'].dtype == 'int32'

    # => try using of dictionary instead of DF

    # create feature columns
    emb_dim_fm = 64
    fixlen_feature_columns = [SparseFeat('user_id', len(user_ids), embedding_dim=emb_dim_fm)]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('history', vocabulary_size=len(item2idx) + 1,
                                              embedding_dim=emb_dim_fm, embedding_name="hist", group_name="history"),
                                                maxlen=config.hist_max_len, combiner='sum')]
    # Var1: combiner='sum'
    # Var2: combiner='mean'
    # Note: combine those features via concatination NOT mean -> produce feature matrix of shape hist_max_len x emb_dim

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    print("Number of features: {}".format(len(feature_names)))

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)

    #train_model_input = {name: train[name] for name in feature_names}
    #    test_model_input = {name: test[name] for name in feature_names}
    train_model_input = {'user_id': train['user_id'], 'history': train['history']}
    test_model_input = {'user_id': test['user_id'], 'history': test['history']}


    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc", "acc"], )
    model.fit(train_model_input, train[target].values,
              batch_size=32, epochs=10, validation_split=0.0, verbose=2)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data input & output
    parser.add_argument('--data_dir', type=str,
                        default='/home/henning/PycharmProjects/thesis-user-modelling/datasets/dpg/',
                        help='path to the dataset file of format *.json.gz')
    parser.add_argument('--pkl_path', type=str, default='../datasets/books-pickle/', help='path to save pickle files')
    parser.add_argument('--load_pkl_data', type=str, default='/home/henning/PycharmProjects/thesis-user-modelling/datasets/dpg/data_neg_sample.pkl',
                        help='path to pickle file with intermediate results')
    parser.add_argument('--use_dummy_data', type=int, default=0, help='Indicate use of Dummy Data')

    # prep params
    parser.add_argument('--neg_sampling_ratio', type=int, default=2, help='Indicate use of negative sampling')
    parser.add_argument('--hist_max_len', type=int, default=10, help='Number of items in user history')

    # training params
    parser.add_argument('--batch_size', type=int, default=128, help='number of review in one batch for Bert')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='maximum length of review, shorter ones are padded; should be in accordance with the input datasets')

    config = parser.parse_args()

    main(config)
