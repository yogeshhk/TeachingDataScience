# Write code that will take a string and make a zig-zag on given number of rows
# Example: "Paypal is hiring" ie "PAYPALISHIRING" => |/|/| like order
#           P A H N
#           APLSIIG
#           Y I R
#  Output : "PAHNAPLSIIGYIR"

def convert_zigzag(s, num):
    if num == 1:
        return s
    rows_dict = {row: "" for row in range(1, num + 1)}
    current_row_index = 1
    increment_row_index = True # False for down
    for c in s: # each letter:
        rows_dict[current_row_index] += c
        if (current_row_index == 1) or ((current_row_index < num) and increment_row_index):
            current_row_index += 1
            increment_row_index = True
        else:
            current_row_index -= 1
            increment_row_index = False
    converted_string = ""
    for row in range(1, num+1):
        converted_string += rows_dict[row]
    return converted_string
