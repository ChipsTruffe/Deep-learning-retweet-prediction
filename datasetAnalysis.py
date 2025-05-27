import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def import_data( file ):
    train = pd.read_csv(file)
    # only use numeric values
    X = torch.tensor(train.iloc[:, [1,3,4,5,6]].to_numpy(dtype=np.float32,na_value=0))#idc 
    y = torch.tensor(train.iloc[:, 2].to_numpy(dtype=np.float32,na_value=0))
    return X,y





if __name__ == '__main__':
    train = pd.read_csv("/home/maloe/dev/SPEIT/Deep Learning/project/data/train.csv")

    [timestamp, retweet_count, user_verified, user_statuses_count, user_followers_count, user_friends_count] = np.array(train.iloc[:, 1:7])


    toPlot = 'follower'
    if toPlot == 'followers':
        #that's a good one
        plt.scatter(user_followers_count,retweet_count, c=user_verified)
        plt.xscale('log')
        plt.yscale('log')

    if toPlot == 'friends':
        #that's a good one
        plt.scatter(user_friends_count,retweet_count, c=user_verified)
        plt.xscale('log')
        plt.yscale('log')

    if toPlot == 'twtNb':
        plt.scatter(user_statuses_count,retweet_count, c=user_verified)
        plt.xscale('log')
        plt.yscale('log')
    plt.show()