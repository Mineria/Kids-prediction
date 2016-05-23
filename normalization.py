# --*-- coding:utf-8 --*--

def normalize_operator(operator):
    if operator == "+":
        return 1
    elif operator == "-":
        return 2
    elif operator == "/":
        return 3
    elif operator == "*":
        return 1000
    else:
        return 1

def normalize_time(time):
    if time <= 20:
        return 0
    elif time > 20:
        return 1

    # if time < 5:
    #     return 0
    # elif time > 5 and time <= 7:
    #     return 1
    # elif time > 7 and time <= 15:
    #     return 2
    # elif time > 15 and time <= 31:
    #     return 3
    # elif time > 21 and time <= 30:
    #     return 4
    # elif time > 30 and time <= 40:
    #     return 5
    # elif time > 40 and time <= 45:
    #     return 6
    # else:
    #     return 7

def normalize_operands(operand):
    if operand < 50:
        return 0
    elif operand > 50 and operand < 70:
        return 1
    else:
        return 200
    # if operand <= 10:
    #     return 1
    # elif operand >= 10 and operand < 20:
    #     return 2
    # elif operand >= 20 and operand < 30:
    #     return 3
    # elif operand >= 30 and operand < 40:
    #     return 4
    # elif operand >= 40 and operand < 50:
    #     return 5
    # elif operand >= 50 and operand < 60:
    #     return 6
    # elif operand >= 60 and operand < 70:
    #     return 7
    # elif operand >= 70 and operand < 80:
    #     return 8
    # elif operand >= 80 and operand < 90:
    #     return 9
    # elif operand >= 90:
    #     return 10

def server_operation_conversion(operator):
    if operator == 3:
        return 8  #Division |  even more complexity
    elif operator == 2:
        return 7  #Multiplication | more complexity
    elif operator == 1:
        return 2  #Resta
    elif operator == 0:
        return 1  #Sum
    else:
        return 0
