import numpy as np

# a = '''
# 92.3913
# 83.6957
# 88.0435
# '''
# a = np.array(list(map(float, a.split())))
# print(a.shape)
a = [71.1408,69.5781,70.6894]
b = [68.7098,69.6822,70.2032]
c = [71.1061,69.5781,69.8211]
d = [70.1337,69.0398,69.8038]

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