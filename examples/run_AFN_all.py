# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


#10 epoch
#bs = 1024
if __name__ == "__main__":
    # data = pd.read_csv('./criteo_sample.txt')
    
    #选取所有的数据集(因为分类任务计算loss时候需要test有label数据，所以test.txt不可用)
    train_file = '/cfs/user/liweiqin/code/kkcode/bigdata_ex3/DeepCTR-Torch/data/train.txt'
    test_file = '/cfs/user/liweiqin/code/kkcode/bigdata_ex3/DeepCTR-Torch/data/test.txt'
    n_sample = int(45850617)
    
    #测试集没有label，训练集包含label、13个I特征、26个C特征
    train_data = pd.read_csv(train_file,sep = '\t',header=None,names=['label']+['I'+str(i) for i in range(1,14)]+['C'+str(i) for i in range(1,27)],nrows=n_sample)
    #test_data = pd.read_csv(test_file,sep = '\t',header=None,names=['I'+str(i) for i in range(1,14)]+['C'+str(i) for i in range(1,27)],nrows=n_sample)
    print("rows30:",train_data.shape[0])
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    #空缺值处理
    train_data[sparse_features] = train_data[sparse_features].fillna('-1', ) #分类特征空缺填'-1'
    train_data[dense_features] = train_data[dense_features].fillna(0, ) #数值特征空缺填0
    #test_data[sparse_features] = test_data[sparse_features].fillna('-1', )
    #test_data[dense_features] = test_data[dense_features].fillna(0, )

    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    #处理分类特征
    for feat in sparse_features:
        lbe = LabelEncoder()
        train_data[feat] = lbe.fit_transform(train_data[feat])
        #test_data[feat] = lbe.fit_transform(test_data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    train_data[dense_features] = mms.fit_transform(train_data[dense_features])
    # test_data[dense_features] = mms.fit_transform(test_data[dense_features])
    print("1")
    # 2.count #unique features for each sparse field,and record dense feature field name
    #处理数值特征
    '''
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=pd.concat([train_data[feat],test_data[feat]]).max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    '''
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=train_data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    print("2")
    # 3.generate input data for model
    #划分数据集

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train,test = train_test_split(train_data, test_size=0.2, random_state=2022)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    print("3")
    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:4'
    model = AFN(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-4, device=device,dnn_dropout=0.3,l2_reg_linear=1e-4)
    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    history = model.fit(train_model_input, train[target].values, batch_size=4096, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)
    torch.save(model,'run_afn_all_epoch10_drop30_adam_l2-4_lr.h5')
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
