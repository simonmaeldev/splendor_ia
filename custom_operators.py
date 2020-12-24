def add(l1, l2):
    return [a + b for a, b in zip(l1, l2)]

def substract(l1, l2):
    return [a - b for a, b in zip(l1, l2)]

def flatten(l):
    return [item for sublist in l for item in sublist]
