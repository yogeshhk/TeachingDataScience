def is_pretty(num):
    list_of_digits = list(str(num))
    digits = set(list_of_digits)
    for d in digits:
        if list_of_digits.count(d) == 1:
            return False
    return True

def generate_palindrom(num):
    if is_pretty(num):
        list_of_digits = list(str(num))
        sorted_list = sorted(list_of_digits)
        resultleft = []
        resultright = []
        for d in range(-1,-1*len(list_of_digits),-2):
            number = sorted_list[d]
            resultleft = [number] + resultleft
            resultright = resultright + [number]
        result_all = resultright + resultleft 
        final_result = [str(i) for i in result_all]
        final_result = "".join(final_result)
        return final_result
    return "NO"
            
        
result = generate_palindrom(55577)
print(result)