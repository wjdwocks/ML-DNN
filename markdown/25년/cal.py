import numpy as np

# a = '''
# 92.3913
# 83.6957
# 88.0435
# '''
# a = np.array(list(map(float, a.split())))
# print(a.shape)
a = [80.6763,80.8917,95.1003,95.7644,89.2743,89.3240,96.7677,65.6434,100.000]
b = [77.2947,81.3503,93.8657,94.8941,87.1587,90.1824,95.7864,64.7992,98.9130]
c = [79.6296,81.2994,94.8302,91.1517,82.0172,90.8530,95.9307,57.4572,98.9130]
d = [78.1535,77.5796,94.9846,92.2541,85.7073,92.7575,96.2482,63.8526,100.000]

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