if __name__ == '__main__':
    N = int(input())
    result = -1
    for i in range(1, N+1):
        result = i
        temp = i
        while(i != 0):
            temp += i % 10
            i = i // 10
        if temp == N:
            print(result)
            break
    if result == N:
        print(0)