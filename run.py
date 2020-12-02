# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import pandas as pd

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    # 需要训练什么模型就自己给model_name命名，名字采用models包下的名字
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # 从原始数据集合中切分好train 和 dev 数据可以不同
    # 家居

    # 读取有标签数据
    data_train = pd.read_csv('THUCNews/data/data_train.csv', engine="python", encoding="utf_8_sig")
    # 选出要训练的正样本
    select_from_data_train = data_train[data_train['class_label'] == 9]
    # 将正样本的标签刷成1
    select_from_data_train['class_label'] = select_from_data_train['class_label'].apply(lambda x:1)
    # 读取负样本数据集
    data_neg = pd.read_csv('THUCNews/data/data_neg.csv', engine="python", encoding="utf_8_sig")
    # 选取和正样本数量一致的负样本
    select_from_data_neg = data_neg.sample(select_from_data_train.shape[0])
    # 将负样本的标签刷成0
    select_from_data_neg['class_label'] = select_from_data_neg['class_label'].apply(lambda x: 0)
    # 合并数据集
    data_merge = pd.merge(select_from_data_train, select_from_data_neg, how="outer")
    # 打乱数据
    data_merge = data_merge.sample(frac=1).reset_index(drop=True)

    # 切分数据集
    train_set = data_merge.loc[:data_merge.shape[0] * 0.8]
    dev_set = data_merge.loc[data_merge.shape[0] * 0.8:]
    # 将数据保存到模型定义好的路径里面去
    train_set.to_csv('THUCNews/data/train.csv', index=False, header=False)
    dev_set.to_csv('THUCNews/data/dev.csv', index=False, header=False)
    # train_data, dev_data, test_data = build_dataset(config)
    train_data, dev_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    # test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    # train(config, model, train_iter, dev_iter, test_iter)
    train(config, model, train_iter, dev_iter, save_path="THUCNews/saved_dict/"+ model_name + '.ckpt')
