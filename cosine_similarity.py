import math


def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")

    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


a = [1, 0]
b = [0.8, 0.6]
c = [0, 1]

print(cosine_similarity(a, b))  # 0.8
print(cosine_similarity(a, c))  # 0.0
