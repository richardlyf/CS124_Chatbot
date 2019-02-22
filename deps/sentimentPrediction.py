import numpy as np

frozen_weights = "sentiment_weights.npy"
neutral_range = 0.01 # Prediction within +/- 0.1 is considered neutral

class SentimentPredictor:
	"""
	Unpacks stored feature vector which is a list of words. All texts can be binarized using this vector.
	Unpacks bias and weight from logistic regression training
	"""
	def __init__(self):
		zipped = np.load(frozen_weights)
		self.feature_vec = zipped[0]
		self.bias = zipped[1]
		self.weight = zipped[2]
		print(self.feature_vec[:10])
		print(self.bias)
		print(self.weight[:10])

	"""
	Transforms the input text into a dense vector using self.feature_vec
	"""
	def transform(self, text):
		words = set(text.split())
		result = np.array([int(word in words) for word in self.feature_vec])
		result = result[np.newaxis, :]
		print(result)
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
	    if score <= neutral_range and score >= -neutral_range:
	    	k = 0
	    elif score > neutral_range:
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
		x = self.transform(data)
		print(x)
		k = self.predict(x)
		return k

if __name__ == "__main__":
	classifier = SentimentPredictor()
	
	#with open('test2_pos.txt', 'r') as test_file:
	#	data = test_file.read().replace('\n', '')
	data = "do I ever get positive reviews?"
	r = classifier.classify(data)
	print(r)
