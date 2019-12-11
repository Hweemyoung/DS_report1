import csv
import os.path as osp
import pandas as pd

path = {}
path['train'] = osp.join(osp.curdir, 'train.csv')
path['test'] = osp.join(osp.curdir, 'test.csv')

def split_datetime(data, column='timestamp'):
    datetime = pd.to_datetime(data[column])
    data['year'] = datetime.map(lambda x: x.year)
    data['month'] = datetime.map(lambda x: x.month)
    data['dayofweek'] = datetime.map(lambda x: x.dayofweek)
    data['hour'] = datetime.map(lambda x: x.hour)
    data['minute'] = datetime.map(lambda x: x.minute)

    return data

def categorize(data, columns=['holiday', 'weather', 'weather_detail']):
    for column in columns:
        data[column] = pd.Categorical(data[column])
        data['code_' + column] = data[column].cat.codes

    return data

def preprocess(data):
    data = split_datetime(data)
    data = categorize(data)

    return data


def main():
    # training
    training_data = pd.read_csv(path['train'])

    # preprocess
    training_data = preprocess(training_data)

    return training_data

data = main()
