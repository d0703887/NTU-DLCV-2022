import os
import pandas as pd
import numpy as np
import csv

train_label = {row[0]: row[1] for row in csv.reader(open('./hw2_data/digits/usps/train.csv'))}
test_label = {row[0]: row[1] for row in csv.reader(open('./hw2_data/digits/usps/test.csv'))}
val_label = {row[0]: row[1] for row in csv.reader(open('./hw2_data/digits/usps/val.csv'))}
output = pd.read_csv('./output.csv').values.tolist()
count = 0
for i in range(len(output)):
    name = output[i][0]
    pred = output[i][1]
    try:
        label = train_label[name]
        if int(pred) == int(label):
            count += 1
    except:
        try:
            label = val_label[name]
            if int(pred) == int(label):
                count += 1
        except:
            label = test_label[name]
            if int(pred) == int(label):
                count += 1




print(count / len(output))