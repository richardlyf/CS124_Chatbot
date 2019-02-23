import numpy as np
import sys
curr_path = sys.path[0] + '/deps/'

frozen_weights = "sentiment_weights.npy"
neutral_range = 0.01 # Prediction within +/- 0.1 is considered neutral

class SentimentPredictor:
	"""
	Unpacks stored feature vector which is a list of words. All texts can be binarized using this vector.
	Unpacks bias and weight from logistic regression training
        Stores the sentiment passed in that countains mapping from words to positive and negative labels
	"""
	def __init__(self, sentiment):
            zipped = np.load(curr_path + frozen_weights)
            self.feature_vec = zipped[0]
            self.bias = zipped[1]
            self.weight = zipped[2]
            self.sentiment = sentiment

	"""
	Transforms the input text into a dense vector using self.feature_vec
	"""
	def transform(self, text):
            posCount = 0
            negCount = 0
            words = text.split()

            # Counts the number of positive and negative words in the text
            for word in words:
                if word in self.sentiment:
                    if self.sentiment[word] == 'pos':
                        posCount += 1
                    else:
                        negCount += 1

            # Construct feature vector for classification
            words = set(words)
            print("There are positive words: " + str(posCount))
            print("There are negative words: " + str(negCount))
            result = [int(word in words) for word in self.feature_vec]
            result.append(posCount)
            result.append(negCount)
            result = np.array(result)
            result = result[np.newaxis, :]
            return result

	"""
	Compute the sigmoid function for the input here.
	Arguments:
	x -- A scalar or numpy array.
	Return:
	s - sigmoid(x)
	"""
	def sigmoid(self, x):
	    s = 1 / (1 + np.exp(-x))
	    return s

	"""
	Predict what class an input belongs to based on its score.
	If sigmoid(X.W+b) is greater or equal to 0.5, it belongs to
	class 1 (positive class) otherwise, it belongs to class -1
	(negative class). class 0 (neutral class) if is within neutral_range.
	Arguments:
	x -- A scalar or numpy array that is the feature vector
	Return:
	k -- the predicted class
	"""
	def predict(self, x):
	    assert x.shape[0] == 1, "x has the wrong shape. Expected a row vector, got: "+str(x.shape)
	    score = self.sigmoid(np.dot(x, self.weight) + self.bias) - 0.5
	    print ("The score is " + str(score))
	    if score <= 0 and score >= -neutral_range:
	    	k = 0
	    elif score > 0:
	    	k = 1
	    else:
	    	k = -1
	    return k

	"""
        Classify a string of words as either positive (klass =1) or negative (klass =-1) or neutral (klass = 0)
        Arguments:
        words -- A string of words (a single movie review)
        Return:
        k - the predicted class. 1 = positive. 0 = neutral. -1 = negative
        """
	def classify(self, words):
		x = self.transform(words)
		k = self.predict(x)
		return k

if __name__ == "__main__":
	classifier = SentimentPredictor()

	#with open('test2_pos.txt', 'r') as test_file:
	#	data = test_file.read().replace('\n', '')
	data = "I liked the movie Titanic"
	r = classifier.classify(data)
	print(r)
