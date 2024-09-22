vector = []
blue_tot = 0
white_tot = 0

def func(x, y, N):
    global blue_tot
    global white_tot
    global vector
    blue = 0
    white = 0
    for i in range(x, x+N):
        for j in range(y, y+N):
            if vector[i][j] == 1:
                blue += 1
            else:
                white += 1
    if blue == N*N:
        blue_tot += 1
    elif white == N*N:
        white_tot += 1
    else:
        if N == 1:
            return
        func(x, y, N//2)
        func(x + N//2, y, N//2)
        func(x, y + N//2, N//2)
        func(x + N//2, y + N//2, N//2)
        


if __name__ == '__main__':    
    N = int(input())
    for i in range(N):
        _list = input()
        _list = [list(map(int, _list.split(' ')))]
        vector += _list
    func(0, 0, N)
    
    print(white_tot)
    print(blue_tot)