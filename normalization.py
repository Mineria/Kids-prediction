# --*-- coding:utf-8 --*--

def normal_time(time):
    if time < 10:
        return 2
    if time < 20:
        return 1
    else:
        return 0

def normal_operator(operator):
    if operator == "+":
        return 1
    elif operator == "-":
        return 2
    elif operator == "/":
        return 3
    elif operator == "*":
        return 4
    else:
        return 4

def normal_operands(operand):
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
