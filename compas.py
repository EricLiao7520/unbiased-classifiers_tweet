from utils import get_base_dfs, COMPASDataset, create_data_splits, explain_model
from models import DivDisClassifier, BaseClassifier, ModelWrapper, HeadWrapper
from train_bert import train_model, trainDivDis
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn as nn
import torch
from explainer import Explainer
import numpy as np

non_violent, violent = get_base_dfs()
train_columns = [
    "juv_fel_count", "juv_misd_count", "juv_other_count",
    "priors_count", "african-american", "caucasian", "hispanic",
    "other", "asian", "native-american", "less25", "greater45",
    "25to45", "felony", "misdemeanor", "two_years_r"
]
dataDf = violent[train_columns]
data = COMPASDataset(dataDf, "two_years_r")
trainData, unlabelData, _, testData, rawTrain, rawUnlabel, _, testRaw= create_data_splits(data)
dataiter = iter(trainData)
batch = next(dataiter)
train_columns.pop()
inputs, targets = batch
input_dim = int(inputs.shape[1])
# baseModel = BaseClassifier(input_dim, [32, 16, 4],dropout_rate= 0.2, num_classes=2)
# fullTrain = ConcatDataset([rawTrain, rawUnlabel])
# fullLoader = DataLoader(fullTrain, 64, True)
criterion = nn.CrossEntropyLoss()
# baseModel, history = train_model(baseModel, fullLoader, testData, criterion, 15, 0.001)

# explain_model(baseModel, baseModel.num_classes, testRaw, 'Base', input_dim, train_columns)
divDisModel = DivDisClassifier(input_dim)
full_loss = trainDivDis(divDisModel, 15, trainData, unlabelData, testData, criterion)
for head in range(divDisModel.num_heads):
    model_name = "DivDis model head_ {}".format(head)
    model_head = HeadWrapper(divDisModel, head)
    model_head.eval()
    explain_model(model_head, divDisModel.num_classes, testRaw, model_name, input_dim, train_columns)

