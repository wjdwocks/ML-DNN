import numpy as np

# a = '''
# 92.3913
# 83.6957
# 88.0435
# '''
# a = np.array(list(map(float, a.split())))
# print(a.shape)
a = [70.7588,70.4115,70.9498,70.9845,69.3350,69.4912]
b = [70.2205,70.3768,69.5781,71.1408,70.0816,69.8038]
c = [69.4565,70.2900,68.9877,69.8385,70.2032,69.2655]
d = [70.0990,69.8038,70.8804,69.6128,69.6649,69.6301]

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