import numpy as np

# a = '''
# 92.3913
# 83.6957
# 88.0435
# '''
# a = np.array(list(map(float, a.split())))
# print(a.shape)
a = [88.1332,88.5843,88.0291]
b = [89.0007,89.0354,87.4046]
c = [90.3971,90.6859,89.8195]
d = [91.1191,90.6137,89.6751]

# print(np.sqrt(26))
print(f'{np.mean(a):.4f}')
print(f'{np.std(a):.4f}')
print(f'{np.mean(b):.4f}')
print(f'{np.std(b):.4f}')
print(f'{np.mean(c):.4f}')
print(f'{np.std(c):.4f}')
print(f'{np.mean(d):.4f}')
print(f'{np.std(d):.4f}')
#print(f'{np.std(d)/np.sqrt(26):.4f}')