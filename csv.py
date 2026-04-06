import csv

# Read training data
with open(r'C:\Users\ADMIN\Downloads\weather.csv') as f:
    reader = csv.reader(f)
    data = list(reader)

# Remove header
data = data[1:]

print("Training data:")
for row in data:
    print(row)
print("-" * 50)

attr_len = len(data[0]) - 1

# Initialize S and G
S = ['0'] * attr_len
G = [['?'] * attr_len]

print("Initial Hypotheses")
print("S =", S)
print("G =", G)
print("-" * 50)

for row in data:
    if row[-1].lower() == 'yes':  # Positive
        for i in range(attr_len):
            if S[i] == '0':
                S[i] = row[i]
            elif S[i] != row[i]:
                S[i] = '?'

        # Remove inconsistent G
        G = [g for g in G if all(g[i] == '?' or g[i] == S[i] for i in range(attr_len))]

    elif row[-1].lower() == 'no':  # Negative
        new_G = []

        for g in G:
            for i in range(attr_len):
                # Only specialize if attribute matches negative example
                if g[i] == '?':
                    if S[i] != '?' and row[i] != S[i]:
                        new_h = g.copy()
                        new_h[i] = S[i]

                        # Avoid duplicates
                        if new_h not in new_G:
                            new_G.append(new_h)

        # Keep only consistent hypotheses
        G = new_G

print("Final Hypotheses")
print("S =", S)
print("G =", G)
