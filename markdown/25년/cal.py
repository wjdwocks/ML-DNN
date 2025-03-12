import numpy as np

# a = '''
# 92.3913
# 83.6957
# 88.0435
# '''
# a = np.array(list(map(float, a.split())))
# print(a.shape)
a = [70.0642 , 70.8977 , 70.4636]
b = [70.9672 , 70.0122 , 70.0642]
c = [89.3130 , 88.3414 , 88.7578]
d = []

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