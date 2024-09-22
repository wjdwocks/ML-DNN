if __name__ == '__main__':
    str = input()
    str = str + '*'
    length = len(str)
    plus = 0
    minus = 0
    first = 0
    first_minus = False
    for i in range(length):
        if str[i] == '-' or str[i] == '+' or str[i] == '*':
            substring = str[first:i]
            first = i+1
            if first_minus :
                minus = minus + int(substring)
            else:
                plus += int(substring)
            if (not first_minus) and str[i] == '-':
                first_minus = True
    
    print(plus-minus)
            