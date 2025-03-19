import numpy as np

# a = '''
# 92.3913
# 83.6957
# 88.0435
# '''
# a = np.array(list(map(float, a.split())))
# print(a.shape)
a = [71.1408,70.1684,70.5157]
b = [70.1511,70.5331,69.6996]
c = [69.5433,69.7343,70.8804]
d = [70.4636,69.6475,69.8732]

# print(np.sqrt(26))
print(f'{np.mean(a):.4f}')
print(f'{np.std(a)/np.sqrt(26):.4f}')
print(f'{np.mean(b):.4f}')
print(f'{np.std(b)/np.sqrt(26):.4f}')
print(f'{np.mean(c):.4f}')
print(f'{np.std(c)/np.sqrt(26):.4f}')
print(f'{np.mean(d):.4f}')
# print(f'{np.std(d):.4f}')
print(f'{np.std(d)/np.sqrt(26):.4f}')