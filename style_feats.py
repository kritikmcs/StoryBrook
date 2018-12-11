#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:57:07 2018

@author: chantelle
"""

#import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#from nltk.util import ngrams
import en_core_web_sm
nlp = en_core_web_sm.load()


def process(lines):
    feat_vec = []
    bow_1 = []
    bow_2 = []
    bow_3 = []
    bow_4 = []
    bow_5 = []
    boc = []
    #count = 0 
    for line in lines:
        cols = line.split('\t')
        cols[1] = cols[1].replace('!', '.')
        cols[1] = cols[1].replace('?', '.')
        sent = cols[1].split('.')
        target = sent[4]
        translator = str.maketrans('', '', string.punctuation)
        target = target.translate(translator)
        target = target.strip()
        target = target.lower()
        
        chars = []
        for c in target:
            if ((c == " ") and (chars[-1] == " ")):
                pass
            else:
                chars.append(c)
        
        t = nlp(target)
        tags = []
        tokens = []
        for tok in t:
            if (tok.pos_ != "SPACE"):
                tags.append(tok.pos_ )
                tokens.append(str(tok))
        
        #count = count + 1
        
        feat = []
        #Feat 1 - length 
        #if(len(tokens) == 0):
            #print(line)
            #print(count)
        feat.append(len(tokens))
        
        for i in range(len(tags)):
            if(tags[i] == 'NOUN'):
                tokens[i] = 'NOUN'
            if(tags[i] == 'PROPN'):
                tokens[i] = 'PROPN'
            if(tags[i] == 'PRON'):
                tokens[i] = 'PRON'
            if(tags[i] == 'VERB'):
                tokens[i] = 'VERB'
            if(tags[i] == 'ADJ'):
                tokens[i] = 'ADJ'
            if(tags[i] == 'ADV'):
                tokens[i] = 'ADV'
        
        #Feat 2 - unigrams
        feat.append(tokens)
        bow_1.append(tokens)
        
        tokens = ['START'] + tokens + ['END']
        
        #Feat 3 - bigrams
        N=2
        bg=[]
        for i in range(len(tokens)-N+1):
            bg.append(tokens[i] + " " +  tokens[i+1])
        feat.append(bg)
        bow_2.append(bg)
            
        #Feat 4 - trigrams
        N=3
        tg=[]
        for i in range(len(tokens)-N+1):
            tg.append(tokens[i] + " " +  tokens[i+1] + " " +  tokens[i+2])
        feat.append(tg)
        bow_3.append(tg)
            
        #Feat 5 - 4grams
        N=4
        fourg=[]
        for i in range(len(tokens)-N+1):
            fourg.append(tokens[i] + " " +  tokens[i+1] + " " +  tokens[i+2] + " " +  tokens[i+3])
        feat.append(fourg)
        bow_4.append(fourg)
            
        #Feat 6 - 5grams
        N=5
        fiveg=[]
        for i in range(len(tokens)-N+1):
            fiveg.append(tokens[i] + " " +  tokens[i+1] + " " +  tokens[i+2] + " " +  tokens[i+3] + " " +  tokens[i+4])
        '''
        if (len(fiveg) == 0):
            print(line)
            print(target)
            print(tokens)
        '''
        feat.append(fiveg)
        bow_5.append(fiveg)
        
        #Feat 7 - char n-grams
        N=4
        charg=[]
        for i in range(len(chars)-N+1):
            charg.append(chars[i] + " " +  chars[i+1] + " " +  chars[i+2] + " " +  chars[i+3])
        feat.append(charg)
        boc.append(charg)
    
        feat_vec.append(feat)
    return bow_1, bow_2, bow_3, bow_4, bow_5, boc, feat_vec






def dummy_fun(doc):
    return doc

def prod_fvec(tfidf_1, tfidf_2, tfidf_3, tfidf_4, tfidf_5, tfidf_c, tr_vec):
    #tfidf_c.vocabulary_ = { k:v for k, v in tfidf_c.vocabulary_.items() if (v>4) }
    #count  = 0
    maxi = 0
    
    embed_size = (len(tfidf_1.vocabulary_) + len(tfidf_2.vocabulary_) +
                len(tfidf_3.vocabulary_) + len(tfidf_4.vocabulary_) + len(tfidf_c.vocabulary_))
    grams = np.ndarray((len(tr_vec), embed_size))
    
    for i, fv in enumerate(tr_vec):
        '''
        #Removing those features with freq less than 5
        for g in fv[1]:
            if g not in tfidf_1.vocabulary_:
                fv[1].remove(g)
                
        for g in fv[2]:
            if g not in tfidf_2.vocabulary_:
                fv[2].remove(g)
         
        for g in fv[3]:
            if g not in tfidf_3.vocabulary_:
                fv[3].remove(g)
        
        for g in fv[4]:
            if g not in tfidf_4.vocabulary_:
                fv[4].remove(g)
        
        for g in fv[5]:
            if g not in tfidf_5.vocabulary_:
                fv[5].remove(g)
        
        for g in fv[6]:
            if g not in tfidf_c.vocabulary_:
                fv[6].remove(g)
        '''
        feat_1 = fv[1]
        feat_2 = fv[2]
        feat_3 = fv[3]
        feat_4 = fv[4]
        #feat_5 = fv[5]
        feat_6 = fv[6]
        #Since keeps disappearing - have to keep removing 1
        fv.remove(fv[1])
        fv.remove(fv[1])
        fv.remove(fv[1])
        fv.remove(fv[1])
        fv.remove(fv[1])
        fv.remove(fv[1])
       
        #mat = tfidf_1.transform([feat_1])
        clen = len(tfidf_1.vocabulary_)
        #print(tfidf_1.transform([feat_1]).toarray().shape)
        grams[i, :clen] = tfidf_1.transform([feat_1]).toarray()
        
        #for i in range(len(feat_1)):
        #    fv.append(mat[i])
        #fv.append(mat)
        #print(mat.todense())
        grams[i, clen: clen + len(tfidf_2.vocabulary_)] = tfidf_2.transform([feat_2]).toarray()
        clen = clen + len(tfidf_2.vocabulary_)
        #mat = tfidf_2.transform([feat_2])
        #print(mat.shape)
        #for i in range(len(feat_2)):
        #    fv.append(mat[i])
        #fv.append(mat)
        #print(mat.todense())
        grams[i, clen :clen + len(tfidf_3.vocabulary_) ] = tfidf_3.transform([feat_3]).toarray()  
        clen = clen + len(tfidf_3.vocabulary_)
        #mat = tfidf_3.transform([feat_3])
        #print(mat.shape)
        #for i in range(len(feat_3)):
        #    fv.append(mat[i])
        #fv.append(mat)
        #print(mat.todense())
        grams[i, clen :clen + len(tfidf_4.vocabulary_) ] = tfidf_4.transform([feat_4]).toarray() 
        clen = clen + len(tfidf_4.vocabulary_)    
        #mat = tfidf_4.transform([feat_4])
        #print(mat.shape)
        #for i in range(len(feat_4)):
        #    fv.append(mat[i])
        #fv.append(mat)
        #print(mat.todense())
        grams[i, clen :clen + len(tfidf_c.vocabulary_) ] = tfidf_c.transform([feat_6]).toarray()  
        clen = clen + len(tfidf_c.vocabulary_)   
        #count = count+1
        #print(count)
        '''
        mat = tfidf_5.transform(feat_5)
        for i in range(len(feat_5)):
            fv.append(mat[i])
        '''
        
        #mat = tfidf_c.transform([feat_6])
        #print(mat.shape)
        #for i in range(len(feat_6)):
        #    fv.append(mat[i])
        #fv.append(mat)
        #print(mat.todense())
        if (fv[0] > maxi):
            maxi = fv[0]
     
    for fv in tr_vec:
        fv[0] = fv[0]/maxi
    return grams

def preproc_single(target):
    translator = str.maketrans('', '', string.punctuation)
    target = target.translate(translator)
    target = target.strip()
    target = target.lower()    
    chars = []
    for c in target:
        if ((c == " ") and (chars[-1] == " ")):
            pass
        else:
            chars.append(c)   
    t = nlp(target)
    tags = []
    tokens = []
    for tok in t:
        if (tok.pos_ != "SPACE"):
            tags.append(tok.pos_ )
            tokens.append(str(tok))
    feat = []
    feat.append(len(tokens))    
    for i in range(len(tags)):
        if(tags[i] == 'NOUN'):
            tokens[i] = 'NOUN'
        if(tags[i] == 'PROPN'):
            tokens[i] = 'PROPN'
        if(tags[i] == 'PRON'):
            tokens[i] = 'PRON'
        if(tags[i] == 'VERB'):
            tokens[i] = 'VERB'
        if(tags[i] == 'ADJ'):
            tokens[i] = 'ADJ'
        if(tags[i] == 'ADV'):
            tokens[i] = 'ADV'  
    #Feat 2 - unigrams
    feat.append(tokens)
    tokens = ['START'] + tokens + ['END']
    
    #Feat 3 - bigrams
    N=2
    bg=[]
    for i in range(len(tokens)-N+1):
        bg.append(tokens[i] + " " +  tokens[i+1])
    feat.append(bg)
        
    #Feat 4 - trigrams
    N=3
    tg=[]
    for i in range(len(tokens)-N+1):
        tg.append(tokens[i] + " " +  tokens[i+1] + " " +  tokens[i+2])
    feat.append(tg)
        
    #Feat 5 - 4grams
    N=4
    fourg=[]
    for i in range(len(tokens)-N+1):
        fourg.append(tokens[i] + " " +  tokens[i+1] + " " +  tokens[i+2] + " " +  tokens[i+3])
    feat.append(fourg)
        
    #Feat 6 - 5grams
    N=5
    fiveg=[]
    for i in range(len(tokens)-N+1):
        fiveg.append(tokens[i] + " " +  tokens[i+1] + " " +  tokens[i+2] + " " +  tokens[i+3] + " " +  tokens[i+4])
    feat.append(fiveg)
 
    #Feat 7 - char n-grams
    N=4
    charg=[]
    for i in range(len(chars)-N+1):
        charg.append(chars[i] + " " +  chars[i+1] + " " +  chars[i+2] + " " +  chars[i+3])
    feat.append(charg)
    return feat

def prod_ng_single(tfidf_1, tfidf_2, tfidf_3, tfidf_4, tfidf_5, tfidf_c, feat_vec):
    maxi = 0 
    embed_size = (len(tfidf_1.vocabulary_) + len(tfidf_2.vocabulary_) +
                len(tfidf_3.vocabulary_) + len(tfidf_4.vocabulary_) + len(tfidf_c.vocabulary_))
    grams = np.ndarray((1, embed_size))
    i=0
    fv = feat_vec
    feat_1 = fv[1]
    feat_2 = fv[2]
    feat_3 = fv[3]
    feat_4 = fv[4]
    feat_6 = fv[6]
    #Since keeps disappearing - have to keep removing 1
    fv.remove(fv[1])
    fv.remove(fv[1])
    fv.remove(fv[1])
    fv.remove(fv[1])
    fv.remove(fv[1])
    fv.remove(fv[1])
   
    clen = len(tfidf_1.vocabulary_)
    grams[i, :clen] = tfidf_1.transform([feat_1]).toarray()
    grams[i, clen: clen + len(tfidf_2.vocabulary_)] = tfidf_2.transform([feat_2]).toarray()
    clen = clen + len(tfidf_2.vocabulary_)
    grams[i, clen :clen + len(tfidf_3.vocabulary_) ] = tfidf_3.transform([feat_3]).toarray()  
    clen = clen + len(tfidf_3.vocabulary_)
    grams[i, clen :clen + len(tfidf_4.vocabulary_) ] = tfidf_4.transform([feat_4]).toarray() 
    clen = clen + len(tfidf_4.vocabulary_)    
    grams[i, clen :clen + len(tfidf_c.vocabulary_) ] = tfidf_c.transform([feat_6]).toarray()  
    clen = clen + len(tfidf_c.vocabulary_)   
    if (fv[0] > maxi):
        maxi = fv[0]
     
    for fv in tr_vec:
        fv[0] = fv[0]/maxi
    return grams

with open("validation_double.txt", "r") as f:
    lines = f.readlines()
    tr_1, tr_2, tr_3, tr_4, tr_5, trc, tr_vec = process(lines)
    

with open("test_double.txt", "r") as f:
    lines = f.readlines()
    te_1, te_2, te_3, te_4, te_5, tec, te_vec = process(lines)
    
tfidf_1 = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None
        )    
tfidf_1.fit(tr_1+te_1)

tfidf_2 = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None
    )    
tfidf_2.fit(tr_2+te_2)

tfidf_3 = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None
    )    
tfidf_3.fit(tr_3+te_3)

tfidf_4 = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None
    )    
tfidf_4.fit(tr_4+te_4)

tfidf_5 = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None
    )    
tfidf_5.fit(tr_5+te_5)

tfidf_c = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None
    )    
tfidf_c.fit(trc+tec)

#full train vector
train_sty_vec = prod_fvec(tfidf_1, tfidf_2, tfidf_3, tfidf_4, tfidf_5, tfidf_c, tr_vec)
test_sty_vec = prod_fvec(tfidf_1, tfidf_2, tfidf_3, tfidf_4, tfidf_5, tfidf_c, te_vec)

def generate_train():
    with open('train.pkl', 'wb') as f:
        pickle.dump(train_sty_vec, f)
        
def generate_test():
    with open('test.pkl', 'wb') as f:
        pickle.dump(test_sty_vec, f)

def get_style_features(sentence):
    
    feat_vec = preproc_single(sentence)
    single_style_vec = prod_ng_single(tfidf_1, tfidf_2, tfidf_3, tfidf_4, tfidf_5, tfidf_c, feat_vec)
    return single_style_vec[0]

if __name__ == '__init__':
    
    #~~SINGLE~~ train vector
    target = "Chantelle was happy to finish her NLP project"
    feat_vec = preproc_single(target)
    print(feat_vec)
    single_style_vec = prod_ng_single(tfidf_1, tfidf_2, tfidf_3, tfidf_4, tfidf_5, tfidf_c, feat_vec)
    print(single_style_vec.shape)

    #train_sty_vec = final_feat_vectors("validation_double.txt")
    #test_sty_vec = final_feat_vectors("test_double.txt")
    #print(train_sty_vec[0])    
    #import numpy as np
#     import pickle

#     #np.save('trainnp.npy', train_sty_vec)
#     #np.save('testnp.npy', test_sty_vec)
