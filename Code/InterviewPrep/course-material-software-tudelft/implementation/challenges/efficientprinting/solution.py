n = int(input())
fives = 0
i = 5
while i <= n:
    fives = fives + n // i
    i = i * 5
print(fives)

# The following line will cause a Run Error (0/3)
# raise Exception

# The following line will cause a Wrong Answer (1/3)
# print(42)

# The following lines will cause a Timeout (2/3)
# while True:
#     pass
