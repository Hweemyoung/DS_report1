import csv
import torch
# from torch import nn, optim
import os.path as osp
import pandas as pd
import layers, modules, models

path = {}
path['train'] = osp.join(osp.curdir, 'train.csv')
path['test'] = osp.join(osp.curdir, 'test.csv')

class Preprocessor:
    def __init__(self, data):
        self.data = data
        self.data_dict = {}

    def split_datetime(self, column='timestamp'):
        datetime = pd.to_datetime(self.data[column])
        # self.data['year'] = datetime.map(lambda x: x.year)
        self.data['month'] = datetime.map(lambda x: x.month)
        self.data['dayofweek'] = datetime.map(lambda x: x.dayofweek)
        self.data['hour'] = datetime.map(lambda x: x.hour)

        return self.data

    def categorize(self, columns=['holiday', 'weather', 'weather_detail', 'hour', 'month', 'dayofweek']):
        for column in columns:
            self.data[column] = pd.Categorical(self.data[column])
            self.data['code_' + column] = self.data[column].cat.codes

        return self.data

    def drop_columns(self, columns=['holiday', 'weather', 'weather_detail', 'timestamp', 'hour']):
        for column in columns:
            self.data = self.data.drop(column, 'columns')

        return self.data

    def scale(self):
        self.data['clouds_cover'] /= 100
        self.data['rain_in_hour'] /= 100

        return self.data

    def cast_tensors(self):
        columns_categorical = ['code_holiday', 'code_weather', 'code_weather_detail', 'code_month', 'code_dayofweek',
                               'code_hour']
        for column in columns_categorical:
            self.data_dict[column] = torch.LongTensor(self.data[column])

        columns_not_categorical = ['temperature', 'rain_in_hour', 'snow_in_hour', 'clouds_cover', 'traffic_volume']
        for column in columns_not_categorical:
            self.data_dict[column] = torch.unsqueeze(torch.FloatTensor(self.data[column]), 1)

        return self.data_dict

    def split_data_and_label(self):
        label = self.data_dict['traffic_volume']
        del self.data_dict['traffic_volume']

        return self.data_dict, label

    def preprocess(self):
        self.split_datetime()
        self.categorize()
        self.drop_columns()
        self.scale()
        self.cast_tensors()
        data_dict, label = self.split_data_and_label()

        return data_dict, label


def main():
    # training
    training_data = pd.read_csv(path['train'])

    # preprocess
    preprocessor = Preprocessor(training_data)
    data_dict, labels = preprocessor.preprocess()
    predictor = models.Predictor(data_dict, labels)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters())
    predictor.pseudo_train(criterion, optimizer, 3)

main()
