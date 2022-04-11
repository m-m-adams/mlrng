def xorshift128():
    x = 123456789
    y = 362436069
    z = 521288629
    w = 88675123

    def _random():
        nonlocal x, y, z, w
        t = x ^ ((x << 11) & 0xFFFFFFFF)  # 32bit
        x, y, z = y, z, w
        w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
        return w

    return _random


x = 123456789
y = 362436069
z = 521288629
w = 88675123


def xorshift(x, y, z, w):
    t = x ^ ((x << 11) & 0xFFFFFFFF)  # 32bit
    w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
    return w


first_four = [xorshift128() for _ in range(4)]


def generate(n):
    x = 123456789
    y = 362436069
    z = 521288629
    w = 88675123
    outs = []
    for i in range(n):
        w = xorshift(x, y, z, w)
        x, y, z = y, z, w
        outs.append()
