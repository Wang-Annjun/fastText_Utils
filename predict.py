import pandas as pd
import fastText as ft
from fastText import train_supervised
from sklearn.model_selection import StratifiedKFold
import time
import os
import json


class ft_predict_Util(object):
    
    @staticmethod
    def get_structure_data(X, y, file_save_dir, file_name):
        """
        :param X: 训练数据[type:list]
        :param y: 训练数据的标签[type:list]
        :param file_save_dir: 结构化后的文件存储的文件夹路径命名[type:string]
        :param file_name: 结构化文件的命名[type:string]
        :return _file: 结构化后的文件[type:txt]
        """
        train_data = zip(X, y)

        if not os.path.exists(file_save_dir): 
            os.makedirs(file_save_dir)

        __file = file_save_dir + file_name
        with open(__file, 'w') as f:
            for i in train_data:
                f.write(i[0] + ' __label__' + str(i[1]))
                f.write('\n')
        print('%s is generated' % __file)
        return __file

    @staticmethod
    def train_data_prepare(train_data, X_label, y_label, file_save_dir, file_name):
        """
        :param train_data: 源训练数据集[type:csv]
        :param X_label: 训练数据在源数据集的命名[type:string]
        :param y_label: 标签在源数据集的命名[type:string]
        :param file_save_dir: 结构化后的文件存储的文件夹路径命名[type:string]
        :param file_name: 结构化文件的命名[type:string]
        :return _file: 结构化后的文件[type:txt]
        """
        data = pd.read_csv(train_data)
        train_X = data[X_label].tolist()
        train_y = data[y_label].tolist()

        return get_structure_data(train_X, train_y, file_save_dir, file_name)
    
    @staticmethod
    def final_train(train_data, X_label, y_label, file_save_dir, file_name, model_dir, model_name, **kwargs):
        """
        :param train_data: 源训练数据集[type:csv]
        :param X_label: 训练数据在源数据集的命名[type:string]
        :param y_label: 标签在源数据集的命名[type:string]
        :param file_save_dir: 结构化后的文件存储的文件夹路径命名[type:string]
        :param file_name: 结构化文件的命名[type:string]
        :param model_dir: 模型保存的文件夹[type:string]
        :param model_name: 模型的命名
        :parma **kwargs: 模型训练的参数
        :return clf: 利用所有数据训练好的模型
        """
        structure_data = train_data_prepare(train_data, X_label, y_label, file_save_dir, file_name)
        clf = train_supervised(input=structure_data, **kwargs)

        if not os.path.exists(model_dir): 
            os.makedirs(model_dir)
        __model = model_dir + model_name
        clf.save_model('%s.bin'% __model)

        return clf
    
    @staticmethod
    def fasttext_predict(model_path, raw_data, X_label):
        """
        :param model_path: fastText模型的路径[type:bin]
        :parm raw_data: 待预测的数据集[type:csv]
        :param X_label: 待预测的X在数据集的列名
        :return data: 含有预测标签的data[type:dataframe]
        """
        clf = ft.load_model(model_path)
        data = pd.read_csv(raw_data)
        X = data[X_label].tolist()
        y_pred = [i[0].replace('__label__', '') for i in clf.predict(X)[0]]
        data['y_pred'] = y_pred

        return data
