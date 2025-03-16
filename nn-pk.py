import copy

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import ast




# targeted

# filename = '11-03-sim-patients-1'
filename = '03-03pk-sim-patients-logistic'
print(filename)

with open(filename + '.csv', mode ='r') as file:
    data = list(csv.reader(file, delimiter=","))
pre = [[float(y) for y in ast.literal_eval(x)] for x in data[3]]
post = [[float(y) for y in ast.literal_eval(x)] for x in data[4]]
m = len(pre)
print(m)
X = [[pre[i][2], post[i][2], post[i][4]] for i in range(m)]
drug_start_sizes = [float(x) for x in data[1]]
min_sizes = [float(x) for x in data[2]]
y = [1 if a/b < 0.7 else 0 for a,b in zip(min_sizes, drug_start_sizes)]
pre_fcns = [0]*m
pre_slopes = [0]*m
post_fcns = [0]*m
post_slopes = [0]*m
fit_diffs = [0]*m
diffs = [0]*m

for i in range(m):
    pre_coef = np.polyfit([1, 1.5, 2], [pre[i][0], pre[i][2], post[i][0]],1)
    pre_slopes[i] = pre_coef[0]
    pre_fcns[i] = np.poly1d(pre_coef)
    post_coef = np.polyfit([2.25, 2.75], [post[i][1], post[i][3]],1)
    post_slopes[i] = (post[i][4]-post[i][2])/pre[i][2]
    post_fcns[i] = np.poly1d(post_coef)
    diffs[i] = post[i][1] - post[i][0]
    fit_diffs[i] = post_fcns[i](2) - pre_fcns[i](2)


n = len(X[0])






# Binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Convert to 2D PyTorch tensors
# print(X[0])
X = torch.tensor(X, dtype=torch.float32)
# print(X[0])
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define two models
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(n, 180)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(n, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Compare model sizes
# model1 = Wide()
# model2 = Deep()
# print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))  # 11161
# print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))  # 11041

# Helper function to train one model
def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 300   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# define 5-fold cross validation test harness
# kfold = StratifiedKFold(n_splits=5, shuffle=True)
# cv_scores_wide = []
# for train, test in kfold.split(X_train, y_train):
#     # create model, train, and get accuracy
#     model = Wide()
#     acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
#     print("Accuracy (wide): %.2f" % acc)
#     cv_scores_wide.append(acc)
# cv_scores_deep = []
# for train, test in kfold.split(X_train, y_train):
#     # create model, train, and get accuracy
#     model = Deep()
#     acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
#     print("Accuracy (deep): %.2f" % acc)
#     cv_scores_deep.append(acc)

# evaluate the model
# wide_acc = np.mean(cv_scores_wide)
# wide_std = np.std(cv_scores_wide)
# deep_acc = np.mean(cv_scores_deep)
# deep_std = np.std(cv_scores_deep)
# print("Wide: %.2f%% (+/- %.2f%%)" % (wide_acc*100, wide_std*100))
# print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))

# rebuild model with full set of training data
# if wide_acc > deep_acc:
#     print("Retrain a wide model")
#     model = Wide()
# else:
#     print("Retrain a deep model")
#     model = Deep()
model = Deep()
acc = model_train(model, X_train, y_train, X_test, y_test)
print(f"Final model accuracy: {acc*100:.2f}%")

model.eval()
with torch.no_grad():
    # Test out inference with 5 samples
    for i in range(5):
        y_pred = model(X_test[i:i+1])
        print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

    plt.figure(figsize = (8,6))
    # Plot the ROC curve
    y_pred = model(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label = 'NN') # ROC curve = TPR vs FPR
    print('NN-AUC: '+ str(roc_auc_score(y_test, y_pred))[:4])



    #targeted/chemo
    biomarker = post_slopes
    
    # # # biomarker = [(np.poly1d(np.polyfit(post_t[i], post_val[i],1))(post_t[i][0]) - np.poly1d(np.polyfit(pre_t[i], pre_val[i],1))(post_t[i][0]) )/pre_val[i][-1] for i in range(l)] #normalized

    k = min(biomarker)
    biomarker_pred = [y/k for y in biomarker]
    fpr, tpr, thresholds = roc_curve(y, biomarker_pred)
    plt.plot(fpr, tpr, label = r'$V$' ) # ROC curve = TPR vs FPR


    print('V-AUC: '+ str(roc_auc_score(y, biomarker_pred))[:4])
    print(filename)

    # plt.title("Chemotherapy ctDNA ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('nn-' +filename+ '-0.7'+ '-3'+ '.eps', format='eps')
    plt.savefig('nn-' +filename+'-0.7'+ '-3'+'.png', format='png')
    plt.show()
