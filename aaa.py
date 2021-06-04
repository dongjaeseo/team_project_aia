'''
def func(a,b):
    if a >= 1000:
        if b > 8000:
            return 'CASE-01'
        elif b > 500:
            return 'CASE-02'
        elif b >= 50:
            return 'CASE-03'
    
    elif a >= 100:
        if b > 8000:
            return 'CASE-04'
        elif b > 500:
            return 'CASE-05'
        elif b >= 50:
            return 'CASE-06'
    
    elif a >= 10:
        if b > 8000:
            return 'CASE-07'
        elif b > 500:
            return 'CASE-08'
        elif b >= 50:
            return 'CASE-09'

print(func(400,7000))
'''
'''
def func(array):
    number = 0
    for n, i in enumerate(array):
        number += i*(10**(len(array)-n-1))
    
    number += 1
    number = str(number)
    number = [int(num) for num in number]

    return number

print(func([4,5,9,1]))
'''

def func(array):
    
    number = ''
    for i in array:
        number += str(i)
    
    number = int(number)

    number += 1
    number = str(number)
    number = [int(num) for num in number]

    return number

print(func([1,7,9,9]))


#######