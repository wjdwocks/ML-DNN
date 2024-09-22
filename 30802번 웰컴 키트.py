if __name__ == '__main__':
    N = input() # 참가자의 수
    S, M, L, XL, XXL, XXXL = map(int, input().split()) # 각 티셔츠 사이즈당 수
    T, P = map(int, input().split()) # 티셔츠와 펜의 묶음 수
    Shirts = [S, M, L, XL, XXL, XXXL]
    result_s = 0
    for i in Shirts :
        temp = int(i / T) if (i % T == 0) else int(i / T + 1)
        result_s += temp   
    
    print(result_s)
    print(int(sum(Shirts)/ P), sum(Shirts)%P)