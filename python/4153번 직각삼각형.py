def three_max(a, b, c):
    if a > b:
        if a > c:
            return [a, b, c]
        else:
            return [c, a, b]
    else:
        if  b > c:
            return [b, a, c]
        else:
            return [c, a, b]

if __name__ == '__main__':
    a=0
    b=0
    c=0
    while True:
        data = input()
        a, b, c = map(int, data.split(' '))
        if a == 0 and b == 0 and c == 0:
             break
        max_val = three_max(a,b,c)
        if (max_val[0] ** 2 == max_val[1] ** 2 + max_val[2] ** 2):
            print("right")
        else:
            print("wrong")
            
    
    
    