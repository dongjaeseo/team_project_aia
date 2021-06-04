def func(a, b):
    if a >= 1000:
        A = 0
    elif a >= 100:
        A = 3
    elif a >= 10:
        A = 6
    
    if b > 8000:
        B = 1
    elif b > 500:
        B = 2
    elif b >= 50:
        B = 3
    
    C = A + B

    return C

