import numpy as np

# a = '''
# 92.3913
# 83.6957
# 88.0435
# '''
# a = np.array(list(map(float, a.split())))
# print(a.shape)
a = [90.3249,91.1913,89.9639]
b = [90.9747,91.0469,90.7581]
c = []
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