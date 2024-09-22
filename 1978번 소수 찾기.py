import math

def isPrime(num):
    if num <= 1:
        return False
    elif num == 2:
        return True
    max = int(math.sqrt(num))
    for i in range(2, max+1):
        if num % i == 0:
            return False
    
    return True
        
result = 0

if __name__ == '__main__':
    N = int(input())
    data = input()
    data = list(map(int, data.split()))
    for num in data:
        if isPrime(num):
          result += 1

    print(result)