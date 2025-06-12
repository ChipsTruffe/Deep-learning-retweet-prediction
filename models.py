import torch
import torch.nn as nn
import torch.nn.functional as F

def MLP(input_dim, hidden_dims,output_dim):
    
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dims[0]),
        nn.ReLU(),
        )
    for i in range(1,len(hidden_dims)):
        model.add_module("linear_"+str(i),nn.Linear(hidden_dims[i-1],hidden_dims[i]),)
        model.add_module("relu_"+str(i),nn.ReLU())
    model.add_module("output",nn.Linear(hidden_dims[-1],output_dim))
    return model