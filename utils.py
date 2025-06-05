import pandas as pd
import torch
from verstack.stratified_continuous_split import scsplit # pip install verstack

def stringFlatten(strList):
# takes a list elements that can be strings or list of strings or nan or a list of any of these things and returns a string
    if type(strList) != list and pd.isna(strList):
        return ''
    if not( type(strList) == str or type(strList) == list):
        raise(ValueError, "argument contains something else than strings")
    if type(strList) == str:
        return strList
    elif type(strList) == list:
        toReturn = ''
        for elem in strList :
            if stringFlatten(elem) == '':
                continue
            else:
                toReturn += "," + stringFlatten(elem)
        return toReturn
def randomize_model_weights(model):
    for param in model.parameters():
        if param.requires_grad:
            param.data = 0.01*torch.rand_like(param)    

def split_data(data, N_sample = None, train_size = 0.7):

    if N_sample == None:
        N_sample = len(data)
    data = data[:N_sample]

    print(f"Truncated dataset to {N_sample} samples")


    if 'retweet_count' in data.columns:
        X_train, X_test, y_train, y_test = scsplit(data, data['retweet_count'], stratify=data['retweet_count'], train_size=train_size)

        X_train = X_train.drop(['retweet_count'], axis=1)
        X_test = X_test.drop(['retweet_count'], axis=1)

        y_train = torch.from_numpy(y_train.to_numpy()).float()
        y_test = torch.from_numpy(y_test.to_numpy()).float()
    else:
        X_train = data
        X_test = data #flemmmmme
        y_test = 0
        y_train = 0

    #Split data between text and scalar
    X_text_train = X_train[['urls','hashtags','text']]
    X_num_train = X_train[['timestamp', 'user_verified', 'user_statuses_count','user_followers_count','user_friends_count']].to_numpy(dtype = float)
    X_text_test = X_test[['urls','hashtags','text']]
    X_num_test = X_test[['timestamp', 'user_verified', 'user_statuses_count','user_followers_count','user_friends_count']].to_numpy(dtype = float)

    #send to torch

    X_num_train = torch.from_numpy(X_num_train).float()
    X_num_test = torch.from_numpy(X_num_test).float()

    return X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test