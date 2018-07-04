import pandas as pd
import fastText as ft
from fastText import train_supervised
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
import time
import os
import pprint
import warnings
warnings.filterwarnings('ignore')


def split_train_test(data, test_ratio):
    """
    :param data: 训练数据[type:csv]
    :param test_ratio: 测试数据的比例[type:int]
    :return: 训练集和测试集
    """
    df = pd.read_csv(data)
    return train_test_split(df, test_size=test_ratio, random_state=42)


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


def classification_report_output(test_txt, model_path):
    """
    输出classification report
    :param test_txt: 验证数据[type:txt]
    :param model_path: fastText模型的路径
    :return: None
    """
    model = ft.load_model(model_path)
    df = pd.read_csv(test_txt, sep='__label__', engine='python')
    df.columns = ['sentence', 'label']
    y_true = df['label'].tolist()
    y_pred = [i[0].replace('__label__', '') for i in model.predict(df['sentence'].tolist())[0]]
    print(classification_report(y_true, y_pred))


def train_data_prepare(train_data, X_label, y_label, file_save_dir, file_name):
    """
    :param train_data: 源训练数据集[type:dataframe]
    :param X_label: 训练数据在源数据集的命名[type:string]
    :param y_label: 标签在源数据集的命名[type:string]
    :param file_save_dir: 结构化后的文件存储的文件夹路径命名[type:string]
    :param file_name: 结构化文件的命名[type:string]
    :return _file: 结构化后的文件[type:txt]
    """
    train_X = train_data[X_label].tolist()
    train_y = train_data[y_label].tolist()

    return get_structure_data(train_X, train_y, file_save_dir, file_name)


def fasttext_train_valid(train_X, train_y, model_dir, model_name, **kwargs):
    """
    :param train_X: 训练数据[type:txt]
    :param train_y: 验证数据[type:txt]
    :param model_dir: 模型保存的文件夹[type:string]
    :param model_name: 模型的命名
    :parma **kwargs: 模型训练的参数
    :return clf, precision, recall: 返回分类器，precision表现和recall表现
    """
    clf = train_supervised(input=train_X, **kwargs)
    result = clf.test(train_y)
    precision = result[1]
    recall = result[2]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    __model = model_dir + model_name
    clf.save_model('%s.bin' % __model)

    __record = {'train_X': train_X,
                'train_y': train_y,
                'model_path': __model,
                'training_parameter': kwargs,
                'precision': precision,
                'recall': recall}
    pprint.pprint(__record, width=1)
    return clf, precision, recall


def k_fold_validation(train_data, X_label, y_label, k, **kwargs):
    """
    :param train_data: 源训练数据集[type:dataframe]
    :param X_label: 训练数据在源数据集的命名[type:string]
    :param y_label: 标签在源数据集的命名[type:string]
    :param k: K折交叉验证中的K设置[type:int]
    :parma kwargs: 模型训练的参数[type:dict]
    :return avg_precision, avg_recall: 交叉验证中的表现
    """
    X = train_data[X_label]
    y = train_data[y_label]

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    precision_collection = []
    recall_collection = []

    order = 1
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('\n' + '-*' * 10 + '%s Fold' % order + '-*' * 10)
        train_txt = get_structure_data(X_train, y_train, 'KFold_data/', 'data_%s.train' % order)
        test_txt = get_structure_data(X_test, y_test, 'KFold_data/', 'data_%s.valid' % order)
        model_name = 'model_%s' % order
        clf, precision, recall = fasttext_train_valid(train_txt, test_txt, 'model/', model_name, **kwargs)

        classification_report_output(test_txt, 'model/' + model_name + '.bin')

        precision_collection.append(precision)
        recall_collection.append(recall)
        order += 1
    avg_precision = sum(precision_collection) / k
    avg_recall = sum(recall_collection) / k
    print('--' * 20)
    __record = {'avg_precision': avg_precision,
                'avg_recall': avg_recall}
    pprint.pprint(__record, width=1)
    return avg_precision, avg_recall


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

    start_time = time.time()
    csv_training_data = YOUR_CSV_FILE
    train, test = split_train_test(csv_training_data, test_ratio)
    k_fold_validation(train, YOUR_X_Label, YOUR_y_Label, k=5, **params)
    print("----%s Running Seconds -----" % (time.time() - start_time))







