**Story Brook: Predicting Short Story Endings**

###### Chantelle D'Silva, Jasleen Sekhon, Kritik Mathur

We used and modified an example LSTM from Pytorch: https://github.com/pytorch/examples/tree/master/word_language_model

The files in that example that we modified are:
1. main.py - def evaluate(): to also give the LogSoftmax value of the next observed word in the sentence.
2. data.py-  class Dictionary(): to handle unknown words in the test data.

#### Language Model Generation
We use main.py to generate the language model.
It can be run with the command *python main.py* but it is recommended that it be run with some additional parameters, i.e., *python -W ignore main.py --cuda --epochs 10*. This example will suppress the PyTorch deprecation warnings, rum the program on a GPU, and train the model for 10 epochs. 

#### The files we wrote
1. style_feats.py - This file gives the style features for a sentence provided to it.
2. nnlm.py - This file uses the RNNLM to get probabilities of the sentences provided to it.
3. train_lr.py - It trains the Logistic Regression model using the *validation_double.txt* file as training data and *test_double.txt* as the test data.
4. demo.py - It predicts the endings for the stories written in the file stories.txt using the classifier trained by *train_lr.py*/


#### Other files
1. model-50.defhidden.pt - This is the language model that was obtained from *main.py*.
2. lr_model.pk - This is the trained Logistic Regression classifier that was trained in *train_lr.py*.
3. corpus.pk - This stores the dictionaries associated with the vocabulary, e.g., id2word and word2id.
4. prob_cache.pk - This stores the probabilities of already seen sentences so that time is saved while creating train and test data.
5. train.pkl and test.pkl - These are pre-saved style feature vectors of the train and the test set to save time when re-running for the same files


#### Dependencies
1. Python 3.6.5 with libraries like sklearn, numpy, pandas etc. - https://conda.io/miniconda.html
2. Pytorch 0.4.1 with CUDA and cudNN - https://pytorch.org/
3. spaCy - https://spacy.io/

#### System Specifications
- Operating System: Ubuntu 16.04
- Intel Skylake 12 vCPUs (specification unknown)
- 32GB RAM
- NVIDIA Tesla P100