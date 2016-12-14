import numpy as np

p1 = np.array([1.0, 1.0, 1.0])
p2 = np.array([0.2, 0.8, 0.4])
p3 = np.array([0.1, 0.2, 0.3])
p4 = np.array([0.1, 0.1, 0.9])

p = [p1, p2, p3, p4]

print(p1 + p2 + p3 + p4)

for i in range(1, 4):
    print(i)
    total_prob = [p[i]*p[i - 1]]
    bayes = total_prob/np.sum(total_prob)
    print(bayes)
    print(np.sum(bayes))
