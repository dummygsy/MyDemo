'''
This method use Monte Carlo Method to calculte the value of pi
'''
from random import random
def pi(n):
    t = 0
    for i in range(n):
        # Randomly distribute multi points in 1*1 square
        x = random() ** 2
        y = random() ** 2
        # The point (x, y) is inside the circle whose radius is 1
        if x + y <= 1:   
            t += 1
        
        if i == 10**2 - 1:
            print("when n = 10**2, pi = ", t / n * 4) 
        if i == 10**3 - 1:
            print("when n = 10**3, pi = ", t / n * 4) 
        if i == 10**4 - 1:
            print("when n = 10**4, pi = ", t / n * 4)            
        if i == 10**5 - 1:
            print("when n = 10**5, pi = ", t / n * 4)                    
        if i == 10**6 - 1:
            print("when n = 10**6, pi = ", t / n * 4)            
        if i == 10**7 - 1:
            print("when n = 10**7, pi = ", t / n * 4)
            
    return t / n * 4

print("Final Result, pi = ", pi(10**7))
