import random
random.seed(1)
cards = [1,2,3,4,5,6,7,8,9,10,11,12,13]
cards += cards + cards + cards
from random import shuffle
import numpy as np
shuffle(cards)
_LEARNINGRATE = .01

class Player():
	def __init__(self):
		
		self.hand = []
		self.b1 = np.random.randn(3, 16)/100000
		self.b2 = np.random.randn(16, 8)/100000
		self.b3 = np.random.randn(8, 1)/100000
		

	def think(self):
		lastCard = self.hand[-1]
		totalOfCards = sum(self.hand)
		numOfCards = len(self.hand)
		inp = np.array([[lastCard, totalOfCards, numOfCards]])

		o1 = np.matmul(inp, self.b1)
		o2 = np.matmul(o1, self.b2)
		o3 = np.matmul(o2, self.b3)
		print(o3[0][0])

		if o3[0][0] > 0:
			self.hit()
		else:
			self.stay()

	def hit(self):
		self.hand.append(self.dealer.pop())
		if sum(self.hand) >= 21:
			self.score()
			print("scoring")
		else:
			self.think()

	def stay(self):
		self.score()

	def score(self):
		sc = sum(self.hand)
		print(sc)
		if sc > 21:
			adjust = .1 ** (sc)
		elif sc < 21:
			adjust = -.1 ** sc
		elif sc == 21:
			adjust = 0
			self.b1 *= 1.1
			self.b2 *= 1.1
			self.b3 *= 1.1
		self.b1 = np.zeros_like(self.b1) + adjust
		self.b2 = np.zeros_like(self.b2) + adjust
		self.b3 = np.zeros_like(self.b3) + adjust
		self.sc = sc



jim = Player()
scores = []
for i in range(1000):
	d = [1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13]
	shuffle(d)
	jim.hand = []
	jim.hand.append(d.pop())
	jim.hand.append(d.pop())
	jim.dealer = d
	jim.think()
	scores.append(jim.sc)

from matplotlib import pyplot as pp
pp.plot(scores)
pp.show()