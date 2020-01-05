import csv
import numpy as np
import torch
# from torch import nn, optim
import os
from os import path as osp
import pandas as pd
import layers, modules, models

path = {}
path['train'] = osp.join(osp.curdir, 'train.csv')
path['test'] = osp.join(osp.curdir, 'test.csv')


class Preprocessor:
    def __init__(self, data, phase='eval'):
        self.phase = phase
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
        if self.phase == 'train':
            self.data['traffic_volume'] /= 1000

        return self.data

    def cast_tensors(self):
        columns_categorical = ['code_holiday', 'code_weather', 'code_weather_detail', 'code_month', 'code_dayofweek',
                               'code_hour']
        for column in columns_categorical:
            self.data_dict[column] = torch.LongTensor(self.data[column])
        if self.phase == 'train':
            columns_not_categorical = ['temperature', 'rain_in_hour', 'snow_in_hour', 'clouds_cover', 'traffic_volume']
        elif self.phase == 'eval':
            columns_not_categorical = ['temperature', 'rain_in_hour', 'snow_in_hour', 'clouds_cover']
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
        if self.phase == 'train':
            return self.split_data_and_label()
        elif self.phase == 'eval':
            return self.data_dict


def main(phase='eval'):
    # training
    if phase == 'train':
        data = pd.read_csv(path['train'])
        # preprocess
        preprocessor = Preprocessor(data, phase)
        data_dict, labels = preprocessor.preprocess()
    elif phase == 'eval':
        data = pd.read_csv(path['test'])
        # preprocess
        preprocessor = Preprocessor(data, phase)
        data_dict = preprocessor.preprocess()

    predictor = models.Predictor()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(predictor.parameters())

    if phase == 'train':
        # train
        # files in directory 'models'
        list_files = os.listdir(osp.join(osp.curdir, 'models'))
        if list_files:
            model_name = ('%03d' % (int(list_files[-1].split('.')[0]) + 1)) + '.pth'
        else:
            model_name = '000.pth'
        model_path = osp.join(osp.curdir, 'models', model_name)
        print('Save path:', model_path)
        # train
        predictor.train(data_dict, labels, criterion, optimizer, 1)
        # save
        torch.save(predictor.state_dict(), model_path)

    elif phase == 'eval':
        model_name = '001.pth'
        model_path = osp.join(osp.curdir, 'models', model_name)
        fname = model_name.split('.')[0] + '.txt'
        fpath = osp.join(osp.curdir, 'preds', fname)
        # load model
        predictor.load_state_dict(torch.load(model_path))
        # predict
        preds = predictor.eval(data_dict)
        # rescale: times 1000
        preds *= 1000
        np.savetxt(fpath, preds.squeeze().numpy())

main(phase='train')
