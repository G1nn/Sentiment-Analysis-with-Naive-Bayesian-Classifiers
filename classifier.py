import os, string
from string import digits
from nltk.corpus import stopwords
from nltk import bigrams, trigrams
from collections import Counter
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
from nltk.stem import PorterStemmer

#load one file
def get_file(folder, filename):
	file = open(os.path.join(folder, filename), 'r')
	text = file.read()
	file.close()
	return text

#delete punctuation marks and numbers, and then tokenize and stem
def clean(text):
	del_punc = str.maketrans('', '', string.punctuation)
	del_digit = str.maketrans('', '', digits)
	no_punc = text.translate(del_punc) 
	no_digit = no_punc.translate(del_digit)
	tokens = no_digit.split()
	tokens = [stemmer.stem(t) for t in tokens]
	return tokens

#delete stop words and words of length 1 and generate unigrams
def unigram(tokens):	
	tokens = [w for w in tokens if w not in stop_words and len(w) != 1]
	return tokens

#delete stop words and words of length 1 and generate bigrams
def bigram(tokens):
	tokens = [w for w in list(bigrams(tokens)) if w[0] not in stop_words and w[1] not in stop_words and len(w[0]) != 1 and len(w[1]) != 1]
	return tokens

#delete stop words and words of length 1 and generate trigrams
def trigram(tokens):
	tokens = [w for w in list(trigrams(tokens)) if w[0] not in stop_words and w[1] not in stop_words and w[2] not in stop_words and len(w[0]) != 1 and len(w[1]) != 1 and len(w[2]) != 1]
	return tokens

#make a dictionary with tokens as keys and their occurrence as values from a folder of data
def make_dictionary(folder, dictionary):
	tokens = []
	for filename in os.listdir(folder):
		text = get_file(folder, filename)
		text_processed = unigram(clean(text))
		tokens += text_processed
	dictionary.update(tokens)
	dictionary = dictionary.most_common(20000)
	return dictionary

#make a feature matrix and a label array for the dataset
def extract_features(folder):
	files = [f for f in os.listdir(folder)]
	matrix = np.zeros((len(files), len(dictionary)), dtype = int)
	labels = np.zeros(len(files), dtype = int)
	matrixb = np.zeros((len(files), len(dictionary)), dtype = int)
	labelsb = np.zeros(len(files), dtype = int)
	file_id = 0
	token_id = 0
	ct = 0
	for f in files:
		text = unigram(clean(get_file(folder, f)))
		for w in text:
			for i, t in enumerate(dictionary):
				if t[0] == w:
					token_id = i
					matrixb[file_id, token_id] = 1
					matrix[file_id, token_id] = text.count(w)
		if f.startswith('pos'):
			labels[file_id] = 1
			labelsb[file_id] = 1
		file_id = file_id + 1
		ct = ct + 1
		print(file_id)
	return matrix, labels, matrixb, labelsb

stemmer= PorterStemmer()
stop_words = set(stopwords.words('english'))
train_folder = '/home/jinshi/Documents/final_project/dataset/train'
test_folder = '/home/jinshi/Documents/final_project/dataset/test'
dictionary = Counter()
dictionary = make_dictionary(train_folder, dictionary)
#print(dictionary)
train_matrix, train_labels, train_matrix_b, train_labels_b= extract_features(train_folder)
test_matrix, test_labels, test_matrix_b, test_labels_b = extract_features(test_folder)
#print(test_matrix[0])
#print(test_matrix_b[0])
#print(test_labels)


#gaussian
gau = GaussianNB()
gau.fit(train_matrix, train_labels)
pred_labels1 = gau.predict(test_matrix)
accuracy1 = accuracy_score(test_labels, pred_labels1)
print("Accuracy of Gaussian Model: %s" % accuracy1)

#multinomial
mult = MultinomialNB()
mult.fit(train_matrix, train_labels)
pred_labels2 = mult.predict(test_matrix)
accuracy2 = accuracy_score(test_labels, pred_labels2)
print("Accuracy of Multinomial Model: %s" % accuracy2)

#bernoulli
bern = BernoulliNB()
bern.fit(train_matrix_b, train_labels_b)
pred_labels3 = bern.predict(test_matrix_b)
accuracy3 = accuracy_score(test_labels_b, pred_labels3)
print("Accuracy of Bernoulli Model: %s" % accuracy3)
