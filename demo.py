print('Loading up everything. Hold on a minute....')
import os
import pickle
from nnlm import probability_of
from style_feats import get_style_features
from sklearn import preprocessing
import numpy as np

class Story:
    
    def __init__(self, story_sentences, option_1, option_2):
        self.sentences = story_sentences
        self.options = [option_1, option_2]
        
def get_probability_features(story_sentences, option):
    
    ending_prob = probability_of(option)
    conditional_prob = probability_of(option, given_sent=' '.join(story_sentences))
    return np.asarray([ending_prob, conditional_prob, conditional_prob - ending_prob])

model_path = 'lr_model.pk'
stories_path = 'stories.txt'

assert os.path.exists(model_path)
assert os.path.exists(stories_path)

classifier = pickle.load(open(model_path, 'rb'))

with open(stories_path) as spf:
    stories = []
    answers = []
    for line in spf.readlines():
        stuff = line.split(',')
        stories.append(Story(stuff[1:5], stuff[5], stuff[6]))
        answers.append(int(stuff[-1].strip()))
        
print()
for i, test_story in enumerate(stories):
    print('Story', i + 1, ' '.join(test_story.sentences))
    for j, option in enumerate(test_story.options):
        print('Option', j+1, option)
    print()
    
    labels = []
    class_probabilities = []

    for option in test_story.options:
        probabilities = get_probability_features(test_story.sentences, option)
        print('P(%s)' %(option), probabilities[0])
        print('P(%s|story)' %(option), probabilities[1])
        print()
        style_features = get_style_features(option)
        features = np.concatenate((probabilities, style_features))
        scaled_features = preprocessing.scale(features).reshape(1, -1)
        labels.append(classifier.predict(scaled_features))
        class_probabilities.append(classifier.predict_proba(scaled_features))

    
    if labels[0] == labels[1]:
        if labels[0] == 1:
            if class_probabilities[0][0][1] < class_probabilities[1][0][1]:
                correct = 2
            else:
                correct = 1
        else:
            if class_probabilities[0][0][0] < class_probabilities[1][0][0]:
                correct = 1
            else:
                correct = 2
    elif labels[0] == 1:
        correct = 1
    else:
        correct = 2
    
    print('Predicted Ending is:', test_story.options[correct - 1])
    print('Correct Ending is:', test_story.options[answers[i] - 1])
    print()
    print()