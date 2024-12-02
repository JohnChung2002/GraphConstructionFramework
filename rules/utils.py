from functools import wraps
from itertools import chain, combinations

"""
Decorator to execute specific rules based on the rule list.

Ensures the rule is only run if the rule is in the rule set.
"""

def rule_wrapper(rules_to_apply):
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            rule_set = set(kwargs.get("rule_set", []))  # Convert to set to ensure compatibility
            if rules_to_apply & rule_set:
                return func(*args, **kwargs)
        return wrapped
    return decorator

# Function to get all subsets of a given set
def all_subsets(rule_list):
    # Generate all possible subsets
    return list(chain.from_iterable(combinations(rule_list, r) for r in range(1, len(rule_list) + 1)))