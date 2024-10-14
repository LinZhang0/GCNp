import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from sklearn.model_selection import train_test_split
import math
import torch.nn.functional as F

dataset = 'helpdesk'
numprefix = 9
lr_value = 1e-03
num_runs=5
num_epochs_gcn=1000

path = '../Data/{}.csv'.format(dataset)
save_folder='Results/{}'.format(dataset)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = pd.read_csv(path)
num_nodes = len(data['ActivityID'])
num_varieties = data['ActivityID'].nunique()


def get_splits(num_nodes, n_test = 1/3, n_valid = 0.5):
    train_idx,valid_idx = train_test_split(list(range(num_nodes)),test_size = n_test, shuffle = True)
    train_idx,test_idx = train_test_split(train_idx,test_size=n_valid,shuffle=True)
    return train_idx,valid_idx,test_idx

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class EventPredictor_mlp(torch.nn.Module):
    def __init__(self,num_node,num_features):
        super(EventPredictor_mlp,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_node * num_features,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256,num_node+1),
        )

    def forward(self, x):
        x = torch.flatten(x)
        x = self.mlp(x)
        return x

train_idx,valid_idx,test_idx = get_splits(num_nodes)


class GCNConv(torch.nn.Module):
    def __init__(self, num_nodes, num_features, out_channels):
        super(GCNConv, self).__init__()

        self.in_channels = num_features
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(num_features, out_channels))
        self.bias = Parameter(torch.Tensor(num_nodes,out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj):
        x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x)
        if self.bias is not None:
            return x + self.bias

class GCNConv_mlp(torch.nn.Module):
    def __init__(self, num_nodes, num_features, out_channels):
        super(GCNConv_mlp, self).__init__()

        self.in_channels = num_features
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(num_features, out_channels))
        self.bias = Parameter(torch.Tensor(num_nodes,out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        x = torch.mm(x, self.weight)  # GraphConvolution forward。input*weight
        if self.bias is not None:
            return x + self.bias


class EventPredictor_gcn(torch.nn.Module):
    def __init__(self,num_nodes, numprefix):
        super(EventPredictor_gcn,self).__init__()
        self.layer1 = GCNConv(num_nodes, numprefix, out_channels=32)
        self.layer2 = GCNConv(num_nodes, 32, out_channels=32)
        self.layer3 = GCNConv_mlp(num_nodes, 32, out_channels=32)

    def forward(self, x, adj):
        x = F.relu(self.layer1(x,adj))
        x = self.layer2(x,adj)
        # x = self.layer3(x)
        return F.log_softmax(x, dim=1)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),  dtype=np.int32)
    return labels_onehot


import scipy.sparse as sp
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

df=data
lastcase = ''
feature2 = []
feature1 = torch.tensor(np.zeros((num_nodes,16)))
feature11 = []
linear = torch.nn.Linear(numprefix * 1, 16)

for i,row in df.iterrows():
    if row[0] != lastcase:
        lastcase = row[0]
        feature2 = []

    feature = row[1]
    feature2.append(int(feature))

    if len(feature2) <= numprefix:
        feature4 = feature2
        feature4 = [0] * (numprefix - len(feature4)) + feature4
    else:
        feature4 = feature2[-numprefix:]
    feature11.append(feature4)
    feature5 = torch.FloatTensor(feature4)
    feature6 = linear(feature5)
    feature1[i]=feature6

numid = 0
link = []
link_above = []
lastcase = ''
numid1 = [0]
label2 = ''
label1 = []

for i,row in df.iterrows():
    if row[0] != lastcase:
        lastcase = row[0]
        label2 = 0
    elif row[0] == lastcase:
        link_above = [numid, numid - 1]
        label2 = row[1]
    label1.append(label2)
    numid += 1
    numid1.append(numid)

del numid1[-1]
del label1[0]
label1.append(0)


for i in numid1:
    for j in numid1:
        if feature11[i] == feature11[j]:
            if i > j:
                link_prefix = [i, j]
                link.append(link_prefix)

labels = encode_onehot(label1)
idx = np.array(numid1, dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.array(link, dtype=np.int32)
print(edges_unordered)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
adj = normalize(adj + sp.eye(adj.shape[0]))
adj = sparse_mx_to_torch_sparse_tensor(adj)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

#3.构建模型
embedding_mlps =feature1.to (torch.float32)
label = torch.tensor(label1)
# label = label1
for run in range(num_runs):
    best_loss = 0
    print("Run:{},Learning Rate:{}".format(run+1,lr_value))
    model_gcn = EventPredictor_gcn(num_nodes, 16)
    optimizer = torch.optim.Adam(model_gcn.parameters(),lr=lr_value)
    model_gcn = model_gcn.to(device)
    adj = adj.to(device)
    epochs_plt = []
    train_acc_plt = []
    train_loss_plt = []
    valid_loss_plt = []
    valid_acc_plt = []

    for epochs in range(num_epochs_gcn):
        model_gcn.train()
        training_loss = 0

        test_predictions = list()
        train_loss_2_s = 0

        optimizer.zero_grad()
        train_prediction = model_gcn(embedding_mlps,adj)
        train_prediction = train_prediction.to('cpu')
        train_loss = nn.CrossEntropyLoss()(train_prediction[train_idx], label[train_idx].to(torch.long))
        train_loss.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            model_gcn.eval()
            optimizer.zero_grad()
            valid_prediction = model_gcn(embedding_mlps, adj)
            valid_prediction = valid_prediction.to('cpu')
            valid_loss = nn.CrossEntropyLoss()(valid_prediction[valid_idx], label[valid_idx].to(torch.long))

        if(epochs==0):
            best_loss = valid_loss
            torch.save(model_gcn,'{}/Event_Predictor_{}_prefix{}_{}_run{}.pt'.format(save_folder,dataset,numprefix,lr_value,run+1))

        if(valid_loss < best_loss):
            torch.save(model_gcn,'{}/Event_Predictor_{}_prefix{}_{}_run{}.pt'.format(save_folder,dataset,numprefix,lr_value,run+1))
            best_loss = valid_loss


        with torch.no_grad():
            model_gcn.eval()
            optimizer.zero_grad()
            test_prediction = model_gcn(embedding_mlps, adj)
            test_prediction = test_prediction.to('cpu')
            test_loss =nn.CrossEntropyLoss()(test_prediction[test_idx], label[test_idx].to(torch.long))

        train_acc = accuracy(train_prediction[train_idx],label[train_idx])
        valid_acc = accuracy(valid_prediction[valid_idx],label[valid_idx])
        test_acc  = accuracy(test_prediction[test_idx]  ,label[test_idx] )

        print('Epoch:{},Loss of training:{},Training accuracy:{},Loss of val:{},Val accuracy:{},Loss of test:{},Train accuracy:{}'.format(epochs,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc))
        epochs_plt.append(epochs+1)
        train_acc_plt.append(train_acc)
        train_loss_plt.append(training_loss)
        valid_acc_plt.append(valid_acc)
        valid_loss_plt.append(valid_loss)

    test_predictions = list()
    num_test = 0
    model_gcn = torch.load('{}/Event_Predictor_{}_prefix{}_{}_run{}.pt'.format(save_folder,dataset,numprefix,lr_value,run+1))
    model_gcn.eval()

    optimizer.zero_grad()
    model_gcn.eval()
    optimizer.zero_grad()
    test_prediction = model_gcn(embedding_mlps, adj)
    test_prediction = test_prediction.to('cpu')
    test_loss = nn.CrossEntropyLoss()(test_prediction[test_idx], label[test_idx].to(torch.long))
    test_acc =accuracy(test_prediction[test_idx],label[test_idx] )
    print("\033[31;40mresult {} result：\nTest loss:{},Test accuracy:{}\n\033[0m".format(run+1,test_loss,test_acc))

    filepath = '{}/Accuracy_{}_{}_run{}.txt'.format(save_folder,dataset,lr_value,run+1)
    with open(filepath,'w') as file:
        for item in zip(epochs_plt,train_acc_plt,train_loss_plt,valid_acc_plt,valid_loss_plt):
            file.write('{}\n'.format(item))
        file.write('result {} result：\nTest loss:{},Test accuracy:{}'.format(run+1,test_loss,test_acc))