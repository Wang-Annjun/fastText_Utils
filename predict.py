import pandas as pd
import fastText as ft
from fastText import train_supervised
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
import time
import os
import json



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
    with open(__file, 'w', encoding='utf8') as f:
        for i in train_data:
            f.write(i[0] + ' __label__' + str(i[1]))
            f.write('\n')
    print('%s is generated' % __file)
    return __file

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

def split_train_test(data, test_ratio):
    """
    :param data: 训练数据[type:dataframe]
    :param test_ratio: 测试数据的比例[type:int]
    :return: 训练集和测试集
    """
    return train_test_split(data, test_size=test_ratio, random_state=42)

def classification_report_output(test_csv, clf):
    df = pd.read_csv(test_csv)
    df = df[['cut', 'label']]
    df.columns = ['sentence', 'label']
    y_true = df['label'].tolist()
    y_pred = [int(i[0].replace('__label__', '')) for i in clf.predict(df['sentence'].tolist())[0]]
    print(classification_report(y_true, y_pred))


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

    classification_report_output(test, clf)

    return clf

def fasttext_predict(model_path, raw_data, X_label):
    """
    :param model_path: fastText模型的路径[type:bin]
    :parm raw_data: 待预测的数据集[type:csv]
    :param X_label: 待预测的X在数据集的列名
    :return data: 含有预测标签的data[type:dataframe]
    """
    clf = ft.load_model(model_path)
    data = pd.read_excel(raw_data)
    X = data[X_label].tolist()
    y_pred = [i[0].replace('__label__', '') for i in clf.predict(X)[0]]
    data['y_pred'] = y_pred

    return data

if __name__ == '__main__':
    params = {
            'lr': YOUR_lr,
            'epoch': YOUR_epoch,
            'wordNgrams': YOUR_wordNgrams,
            'dim': YOUR_dim,
            'minCount': YOUR_minCount,
            'minn': YOUR_minn,
            'maxn': YOUR_maxn,
            'bucket': YOUR_bucket,
            'loss': YOUR_loss
        }
      
    model = final_train(YOUR_CSV_FILE.csv, YOUR_X_Label, YOUR_y_Label, 'data/', YOUR_FILE_NAME, 'model/', YOUR_MODEL_NAME, **params)

