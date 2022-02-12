from random import randint


def r(x, depth=0):
    if x < 1023:
        return [x] + r(2 * x + 1, depth + 1) + r(2 * x + 2, depth + 1)
    print(depth, x)
    return []


lst = r(0)
