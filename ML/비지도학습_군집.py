import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')

print(fruits.shape) # (300, 100, 100)

print(fruits[0][0][:])
print(fruits[0, 0, :])
print(fruits[0][0])
print('---------------------------------')

color_img = np.random.rand(100, 100, 3)

plt.imshow(color_img)
plt.title('Random Color Image')
plt.show()

apple = fruits[0:100, :, :]
apple = apple.reshape(-1, 10000)
pineapple = fruits[100:200, :, :]
pineapple = pineapple.reshape(-1, 10000)
banana = fruits[200:300, :, :]
banana = banana.reshape(-1, 10000)

plt.figure(2)
plt.hist(np.mean(apple, axis=1))
plt.hist(np.mean(pineapple, axis=1))
plt.hist(np.mean(banana, axis=1))
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

# fig, axis = plt.subplots(1, 3, figsize=(12,4)) # figsize는 fig(전체 그림)의 크기를 의미함.
# axis[0].bar(range(10000), np.mean(apple, axis=0))
# axis[1].bar(range(10000), np.mean(pineapple, axis=0))
# axis[2].bar(range(10000), np.mean(banana, axis=0))
# plt.show()

apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

