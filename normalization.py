# --*-- coding:utf-8 --*--

def normalize_operator(operator):
    if operator == "/":
        return 4 #  even more complexity
    elif operator == "*":
        return 3 #  more complexity
    elif operator == "-":
        return 2
    elif operator == "+":
        return 1
    else:
        return 0

def normalize_time(time):
    return time % 10
    if tim < 5:
        return 1
    else:
        return 0

def server_operation_conversion(operator):
    if operator == 3:
        return 8  #Division |  even more complexity
    elif operator == 2:
        return 7  #Multiplication | more complexity
    elif operator == 1:
        return 2  #Resta
    elif operator == 0:
        return 1  #Sum
    else:
        return 0

def normalize_operands(operand):
    if operand <= 10:
        return 1
    elif operand >= 10 and operand < 20:
        return 2
    elif operand >= 20 and operand < 30:
        return 3
    elif operand >= 30 and operand < 40:
        return 4
    elif operand >= 40 and operand < 50:
        return 5
    elif operand >= 50 and operand < 60:
        return 6
    elif operand >= 60 and operand < 70:
        return 7
    elif operand >= 70 and operand < 80:
        return 8
    elif operand >= 80 and operand < 90:
        return 9
    elif operand >= 90:
        return 10
