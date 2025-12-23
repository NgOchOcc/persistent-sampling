from math_verify import parse, verify

gold = parse("${1,3} \\cup {2,4}$")
answer = parse("${1,2,3,4}$")

# Order here is important!
print(verify(gold, answer))
# >>> True