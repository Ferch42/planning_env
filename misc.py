
def generate_combinations(n, k):
    if k == 1:
        return [[n]]  # Base case: all remaining blocks go into the last box
    combinations = []
    for i in range(n + 1):  # Try placing 0 to n blocks in the current box
        for rest in generate_combinations(n - i, k - 1):
            combinations.append([i] + rest)
    return combinations

#print(generate_combinations(10,4))
