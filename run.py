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

    i = 0
    start_time = time.time()
    print("Loading data...")

    # 读取数据
    data_train = pd.read_csv('THUCNews/data/data_train.csv', engine="python", encoding="utf_8_sig")
    # 划分边界
    boundary1 = int(data_train.shape[0] * i / 5.0)
    boundary2 = int(data_train.shape[0] * (i+1) / 5.0)
    # 取中间的那一份作为验证集
    dev_set = data_train[boundary1:boundary2]
    # 剩下的数据合并为训练集
    train_set = pd.merge(data_train[:boundary1], data_train[boundary2:], how="outer")
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
    train(config, model, train_iter, dev_iter, save_path="THUCNews/saved_dict/"+ model_name + str(i) + '.ckpt')
