import numpy as np

# Define MDP components
states = ['Town', 'Castle', 'Dungeon', 'Market']  # Set of states
actions = ['fight', 'trade']  # Possible actions

discount_factor = 0.9  # Discount factor for future rewards

# Transition probabilities (state, action) -> [(prob, next_state, reward)]
transitions = {
    ('Town', 'fight'): [(1.0, 'Dungeon', 5)],
    ('Town', 'trade'): [(1.0, 'Market', 3)],
    ('Castle', 'fight'): [(1.0, 'Dungeon', 6)],
    ('Castle', 'trade'): [(1.0, 'Market', 4)],
    ('Dungeon', 'fight'): [(1.0, 'Castle', 7)],
    ('Dungeon', 'trade'): [(1.0, 'Market', 2)],
    ('Market', 'fight'): [(1.0, 'Town', 3)],
    ('Market', 'trade'): [(1.0, 'Market', 0)],
}

# Random policy (50% probability of choosing fight or trade in each state)
policy = {
    'Town': {'fight': 0.5, 'trade': 0.5},
    'Castle': {'fight': 0.5, 'trade': 0.5},
    'Dungeon': {'fight': 0.5, 'trade': 0.5},
    'Market': {'fight': 0.5, 'trade': 0.5},
}

# Initialize value function
V = {s: 0 for s in states}

def policy_evaluation(policy, V, theta=0.0001):
    """Evaluate a policy using iterative updates."""
    while True:
        delta = 0
        new_V = V.copy()
        for s in states:
            v = 0
            for a, action_prob in policy[s].items():
                for prob, next_state, reward in transitions[(s, a)]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            new_V[s] = v
            delta = max(delta, abs(V[s] - v))
        V = new_V
        if delta < theta:
            break
    return V

# Perform policy evaluation
V = policy_evaluation(policy, V)

# Print final value function
print("State values after policy evaluation:")
for state, value in V.items():
    print(f"V({state}) = {value:.2f}")