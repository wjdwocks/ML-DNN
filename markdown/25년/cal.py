import numpy as np

a = '''
92.3913
83.6957
88.0435
'''
a = np.array(list(map(float, a.split())))
print(a.shape)
print(f'{np.mean(a):.4f}')
print(f'{np.std(a):.4f}')