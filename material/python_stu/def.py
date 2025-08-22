def add(num1, num2):
    return num1 + num2


if __name__ == '__main__':

    for num in range(1, 3):
        result = add(num, num + 1)
        print("result {}".format(result))