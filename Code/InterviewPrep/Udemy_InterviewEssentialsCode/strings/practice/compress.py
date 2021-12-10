def compress(string):
    compressed = string[0]
    count = 1
    for i in range(1, len(string)):
        if (string[i] == compressed[-1]):
            count += 1
        else:
            compressed += str(count) + string[i]
            count = 1
    return compressed + str(count)

print(compress("aaaabbccddcdbbbaaaac"))