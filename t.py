from collections import Counter
import random
from matplotlib import pyplot as plt


# n = 1000000000
# c = Counter(random.randint(0, 112 - 48) for _ in range(n))
# for i in range(1,11):
#     print ('%2s  %02.10f%%' % (i, c[i] * 100.0 / n))

n = 10000
a = [random.gauss((112-48)//2, (112-48)//6) for _ in range(n)]
b = [i for i in a if i > (112-48)]
print(b)
plt.hist(a, bins=100)
plt.show()