from random import random
def pi(n):
	t = 0
	for _ in range(n):
		x = random() ** 2
		y = random() ** 2
		if x + y <= 1:
			t += 1
	return t / n * 4
print(pi(10**7))
