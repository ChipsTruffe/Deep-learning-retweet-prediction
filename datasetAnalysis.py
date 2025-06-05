import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tokenizer import *
import torch
from nltk.tokenize import TweetTokenizer

""" Provides the import functions if imported, and a whole bunch of ugly analysis stuff if used as main"""

def import_numerical_data( file ):
    train = pd.read_csv(file)
    # only use numeric values
    X = train.iloc[:, [1,3,4,5,6]].to_numpy(dtype=np.float32,na_value=0)#idc 
    y = train.iloc[:, 2].to_numpy(dtype=np.float32,na_value=0)
    return X,y
def import_data(file):
    train = pd.read_csv(file)
    X_num = train.iloc[:, [1,3,4,5,6]].to_numpy(dtype=np.float32,na_value=0)
    X_text = train.iloc[:,10].to_numpy(dtype=str, na_value='')
    y = train.iloc[:, 2].to_numpy(dtype=np.float32,na_value=0)
    return X_num, X_text ,y 

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)



if __name__ == '__main__':
    train = pd.read_csv("/home/maloe/dev/SPEIT/Deep Learning/project/data/train.csv")
    toDo = 'hashtagStats'
    [timestamp, retweet_count, user_verified, user_statuses_count, user_followers_count, user_friends_count] = np.array(train.iloc[:, 1:7]).transpose()
    
    if toDo == 'textStats':
        for tokenizer in [clean_tokenizer]:
            text_data = train.iloc[:,10].to_numpy().transpose()
            vocab = build_vocab(text_data, tokenizer, float(np.inf))
            distribution = [value[1] for _,value in vocab.items()]
            distribution = np.array(distribution)
            distribution = sorted(distribution,  reverse=True)
            plt.loglog( distribution, label = tokenizer.__name__)
        print(np.sum(distribution[:10000])/np.sum(distribution))
        plt.title("Tokenizer comparison")
        plt.legend()
        #plt.savefig("/home/maloe/dev/SPEIT/Deep Learning/project/TokenizerComp.png")
        plt.show()

    if toDo == 'hashtagStats':
        def clean_and_split_hashtags(hashtag_string):
            """
            Cleans a comma-separated hashtag string into a list of lowercase, stripped hashtags.
            Handles None, empty strings, and extra spaces.
            """
            if pd.isna(hashtag_string) or not str(hashtag_string).strip():
                return []  # Return an empty list for NaN, None, or empty/whitespace-only strings
            
            # Split by comma, strip whitespace from each part, convert to lowercase,
            # and filter out any empty strings that might result from extra commas.
            cleaned_tags = [
                tag.strip().lower() 
                for tag in str(hashtag_string).split(',') 
                if tag.strip()
            ]
            return cleaned_tags
        train['clean_tags'] = train['hashtags'].apply(clean_and_split_hashtags)
        train_exploded = train.explode('clean_tags')
        hashtag_total_retweets = train_exploded.groupby('clean_tags')['retweet_count'].mean()


        text_data = train['hashtags'].to_numpy().transpose()
        vocab = build_vocab(text_data, hashtag_tokenizer, float(np.inf))
        distribution = {text: value[1] for text,value in vocab.items()} #text value and hashtag frequency


        distribution = dict(sorted(distribution.items(), key = lambda item : item[1],  reverse=True)) #sorts by hashtag frequency

        rt_by_hashtag = []
        for target_tag,_ in distribution.items():
            # Convert the target tag to lowercase for lookup, as our aggregated keys are lowercase
            target_tag_lower = target_tag.lower()
            # Use .get() to default to 0 if the hashtag isn't found in our aggregated sums
            rt_by_hashtag.append(hashtag_total_retweets.get(target_tag_lower, 0))
        plt.scatter(range(len(rt_by_hashtag)),rt_by_hashtag)
        plt.yscale('log')
        plt.title("rt number by hashtag")
        #plt.savefig("/home/maloe/dev/SPEIT/Deep Learning/project/URLDistrib.png")
        plt.show()


 
    if toDo == 'followers':
        #that's a good one
        plt.scatter(user_followers_count,retweet_count, c=user_verified)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('number of followers')

    if toDo == 'friends':
        #that's a good one
        plt.scatter(user_friends_count,retweet_count, c=user_verified)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('number of friends')

    if toDo == 'twtNb':
        plt.scatter(user_statuses_count,retweet_count, c=user_verified)
        plt.xscale('log')
        plt.yscale('log')
    plt.ylabel('retweets count')
    plt.legend()
    
    plt.savefig(toDo + '.png')