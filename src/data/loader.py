import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath("..")) #need to do this to find src when launching from notebook
from src.data.pca import library_pca

def normalize(data, mean, std_dev):
    data = data - mean
    data = data / std_dev
    return data


def preprocess(train, val, eval):
    ########### Standardization 
    train_mean = np.nanmean(train, axis=0) #need to exclude NaN values
    train_std_dev = np.nanstd(train, axis=0) #same here
    train = normalize(train, train_mean, train_std_dev)

    #now I will normalize the validation and evaluation sets with the mean and stdd from the test set
    val = normalize(val, train_mean, train_std_dev)
    eval = normalize(eval, train_mean, train_std_dev)

    #lets count how many NaN values we have before we get rid of them
    nan_train = np.sum(np.isnan(train))
    nan_val = np.sum(np.isnan(val))
    nan_eval = np.sum(np.isnan(eval))
    print("Number of NaN values:")
    print(f"Training:    {nan_train} out of {train.size}  -- that is {round((nan_train/train.size) * 100, 4)} percent.")
    print(f"Validation:  {nan_val} out of {val.size}   -- that is {round((nan_val/val.size) * 100, 4)} percent.")
    print(f"Evaluation:  {nan_eval} out of {val.size}  -- that is {round((nan_eval/val.size) * 100, 4)} percent.")
    print(f"Overall:     {nan_eval + nan_train + nan_val} out of {val.size + train.size + eval.size}  -- that is {round(((nan_eval + nan_train + nan_val)/(val.size + train.size + eval.size)) * 100, 4)} percent.")

    #before I can do PCA I need to get rid of NaN values, here is how I will do that
    train = train[~np.isnan(train).any(axis=1)]
    val = val[~np.isnan(val).any(axis=1)]
    eval = eval[~np.isnan(eval).any(axis=1)]

    ########### PCA
    #and by that I mean, go get a PCA model that you can use to transform other data
    pca_er = library_pca(train)
    train = pca_er.transform(train)
    val = pca_er.transform(val)
    eval = pca_er.transform(eval)

    return train, val, eval


def ibrl():
    file_path = '../data/IBRLdata.txt' #NOTE: This may need to back out of the directory with ../ depending on where I run this python file from
    #there are 2313682 lines in this file, each representing a set of readings 
    #BUT we only have full readings until line 2313153

    #I want to make a training dataset, a validation dataset, and an evaluation dataset
    #they will be 70% training, 15% validation, and 15% evaluation
    train_size = int(0.7 * 2313153)
    val_size = int(0.15 * 2313153)
    eval_size = int(0.15 * 2313153) #allows us to not include the end values that don't have readings

    #format:
    #date:yyyy-mm-dd	time:hh:mm:ss.xxx	epoch:int	moteid:int	temperature:real	humidity:real	light:real	voltage:real
    df = pd.read_csv(file_path, delimiter=" ", header=None)
    #only use sensor columns for PCA + SVM (real numbers)
    X = df.iloc[:, -4:].to_numpy()

    #NOTE: I do not think I need the time data... because I am just classifying outliers in the usual values?

    X_train = X[:train_size]
    X_val   = X[train_size:train_size+val_size]
    X_eval  = X[train_size+val_size:train_size+val_size+eval_size+1]

    return X_train, X_val, X_eval


if __name__ == "__main__":
    #load the dataset from the Intel Berkeley Research Lab
    train, val, eval = ibrl()

    #let's do some histogram-based labeling (ha)
    #only look at first thousand vals

    #show 10,000 then go to 60,000 to illustrate issue (!!!!1)
    temperature = train[0:10000, 0] #first column
    #np.arange to get epoch numbers starting from 1
    epochs = np.arange(1, len(temperature) + 1)

    #plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, temperature, marker='o', linestyle='', color='b')
    plt.title('Temperature over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Temperature')
    plt.grid(True)
    plt.show()

    #now humidity!!!
    #show 10,000 then go to 60,000 to illustrate issue (!!!!1)
    humidity = train[0:10000, 1] #first column
    #np.arange to get epoch numbers starting from 1
    epochs = np.arange(1, len(humidity) + 1)

    #plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, humidity, marker='o', linestyle='', color='b')
    plt.title('Humidity over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Humidity')
    plt.grid(True)
    plt.show()

    #standardization and PCA
    train, val, eval = preprocess(train, val, eval)
    
