states = ['Rainy', 'Sunny', 'Cloudy']
actions = ['Umbrella', 'No Umbrella']

transitions = {
    'Rainy': {'Rainy': 0.6, 'Sunny': 0.2, 'Cloudy': 0.2},
    'Sunny': {'Rainy': 0.1, 'Sunny': 0.7, 'Cloudy': 0.2},
    'Cloudy': {'Rainy': 0.3, 'Sunny': 0.5, 'Cloudy': 0.2}
}

def reward(s, a, s_prime):
    if a == 'Umbrella':
        return 1 if s_prime == 'Rainy' else -1
    else:  # No Umbrella
        return -10 if s_prime == 'Rainy' else 1

V = {s: 0 for s in states}
gamma = 0.9  
theta = 1e-6  


while True:
    delta = 0
    V_new = {}
    for s in states:
        max_val = -float('inf')
        for a in actions:
            current_val = 0
            for s_prime, prob in transitions[s].items():
                r = reward(s, a, s_prime)
                current_val += prob * (r + gamma * V[s_prime])
            if current_val > max_val:
                max_val = current_val
        V_new[s] = max_val
        delta = max(delta, abs(V[s] - V_new[s]))
    if delta < theta:
        break
    V = V_new.copy()

policy = {}
for s in states:
    best_action = None
    max_val = -float('inf')
    for a in actions:
        current_val = 0
        for s_prime, prob in transitions[s].items():
            r = reward(s, a, s_prime)
            current_val += prob * (r + gamma * V[s_prime])
        if current_val > max_val:
            max_val = current_val
            best_action = a
    policy[s] = best_action


print("Optimal Policy:")
for s in states:
    print(f"{s}: {policy[s]}")

print("\nOptimal Value Function:")
for s in states:
    print(f"{s}: {V[s]:.4f}")
