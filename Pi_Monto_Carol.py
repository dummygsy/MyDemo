from random import random
def pi(n):
	t = 0
	for i in range(n):
		x = random() ** 2
		y = random() ** 2
		if x + y <= 1:
			t += 1
		
		if i == 10**4 - 1:
			print("when n = 10000, pi = ", t / n * 4)			
		if i == 10**5 - 1:
			print("when n = 100000, pi = ", t / n * 4)					
		if i == 10**6 - 1:
			print("when n = 1000000, pi = ", t / n * 4)			
		if i == 10**7 - 1:
			print("when n = 10000000, pi = ", t / n * 4)
			
	return t / n * 4

print("Final Result, pi = ", pi(10**7))
