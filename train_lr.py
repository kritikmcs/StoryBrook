import os
import csv
import pandas as pd
import numpy as np
from nnlm import probability_of
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import style_feats as feats
import pickle
print('Imported Style Features')


def load_data(file_name):
    f = open(file_name)
    text = f.read()
    f.close()
    return text

if __name__ == '__main__':
    
    model_path = 'lr_model.pk'
    
    validation_set = 'validation_double.txt'
    test_set = 'test_double.txt'
    
    train_style_feats = 'train.pkl'
    test_style_feats = 'test.pkl'
    
    if os.path.exists(train_style_feats):
        train_style = pickle.load(open(train_style_feats, 'rb'))
    else:
        feats.generate_train()
    
    if os.path.exists(test_style_feats):
        test_style = pickle.load(open(test_style_feats, 'rb'))
    else:
        feats.generate_test()
    print('Train and Test Style Features generated')
    

    text = load_data(validation_set)
    with open('csvfile.csv','w') as file:
        for line in text:
            file.write(line)
            
    file = pd.read_csv('csvfile.csv', sep = '\t', header = None, names = ['A','B'])
    l = file['B'].tolist()
    
    for i in l :
        i.replace('!','.')
        
    m= []
    for i in l :
        i=i.strip('.!').split('.')
        m.append(i)
        
    target = file['A']

    
    
    f_test = load_data(test_set)
        
    with open('csvfile_test.csv','w') as file_test:
        for line in f_test:
            file_test.write(line)
        
    file_test = pd.read_csv('csvfile_test.csv', sep = '\t', header = None, names = ['A','B'])
    l_test = file_test['B'].tolist()
    
    m_test= []
    for i in l_test :
        i=i.strip('.!').split('.')
        m_test.append(i)
    
    target_test = file_test['A']
    
    print('Making x')
    x = np.zeros((len(m),3 + train_style.shape[1]))
    for i,story in enumerate(m):
        ending_prob = probability_of(story[-1])
        st = ''.join(story[:4])
        ending_story_prob = probability_of(story[-1], given_sent = st)
        prob = ending_story_prob - ending_prob
        x[i][0] = ending_prob
        x[i][1] = ending_story_prob
        x[i][2] = prob
    
    x[:, 3:] = train_style
    x_scaled = preprocessing.scale(x)
    
    if os.path.exists(model_path):
        print('Using existing model')
        model = pickle.load(open(model_path, 'rb'))
    else:
        print('Training a new model')
        model = LogisticRegressionCV(cv=2, random_state=0, verbose=True, n_jobs=16, max_iter=10000).fit(x_scaled, target)
        pickle.dump(model, open(model_path, 'wb'))

    print('Model Ready')
        
    print('Making x_test')
    x_test = np.zeros((len(m_test),3 + test_style.shape[1]))
    for i,story in enumerate(m_test):
        ending_prob = probability_of(story[-1])
        st = ''.join(story[:4])
        ending_story_prob = probability_of(story[-1], given_sent = st)
        prob = ending_story_prob - ending_prob
        x_test[i][0] = ending_prob
        x_test[i][1] = ending_story_prob
        x_test[i][2] = prob

    print('Scaling features')
    x_test[:, 3:] = test_style
    x_test_scaled = preprocessing.scale(x_test)
    
    target_pred = model.predict(x_test_scaled)
    target_pred_proba = model.predict_proba(x_test_scaled)
    
    pred_proba = []
    for i in target_pred_proba:
        pred_proba.append(i[1])
        
    target_pred_list = target_pred.tolist()
    
    target_pred_list_mod=[]
    for i in range(0, len(target_pred_list), 2):
        if target_pred_list[i] == target_pred_list[i+1]:
            if pred_proba[i] > pred_proba[i+1]:
                target_pred_list_mod.append(1)
                target_pred_list_mod.append(0)
            else:
                target_pred_list_mod.append(0)
                target_pred_list_mod.append(1)
        else:
            target_pred_list_mod.append(target_pred_list[i])
            target_pred_list_mod.append(target_pred_list[i+1]) 

    print('Accuracy on test set', accuracy_score(target_test, target_pred_list_mod))
    
