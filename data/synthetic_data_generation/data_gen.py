import random
import string
# import torch

def randomString(stringLength):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def genTrainingSentTopic(numOfTopic=2, numSample = 20, wordSize = 5, numRandomWordPerSent = 9, randomWordVocSize = 100, distr = [0.2, 0.4, 0.6, 0.8, 1]):
	if (len(distr)!= numOfTopic):
		raise Exception("distr and nomOfTopic Inconsistent")
	randomWordDict = []
	sentList = []
	topicList = []
	topicWordSet = set()
	# create a random word dictionary
	for i in range(randomWordVocSize):
		randomWordDict.append(randomString(wordSize))

	# generate random sentences
	
	for i in range(numSample):
		rand = random.random()
		for topicIdx, cumProb in enumerate(distr):
			if  rand <= cumProb:
				break

		sent = []

		# add the topic word
		topicWordIndOfTopic = random.randint(1,2) # so here we assume each topic at least has 1-3 topic words
		topicWordName = '$'+str(topicIdx)+'.'+str(topicWordIndOfTopic)
		sent.append(topicWordName)
		topicWordSet.add(topicWordName)
		# add the rest numRandomWordPerSent random words by randomly selecting words
		# from the randomWordDict
		for j in range(numRandomWordPerSent):
			sent.append(randomWordDict[random.randint(0,randomWordVocSize-1)])

		sentList.append(sent)
		topicList.append(topicIdx)

	wordVocDict = list(topicWordSet)+randomWordDict
	return (sentList, topicList, wordVocDict, list(topicWordSet), list(randomWordDict))
