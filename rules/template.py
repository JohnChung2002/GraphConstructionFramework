from .utils import rule_wrapper, all_subsets

"""
Since python passes by reference, we can modify the edges dictionary in place

process_rules is the main function that will be called to process the rules
It will generate all possible combinations of rules and call the edge_construction function with the rule set
"""

@rule_wrapper({1})
def rule_1(data, edges, **kwargs):
    # Process Rule 1 here
    pass

def edge_construction(data, rule_list):
    edges = {}
    rule_1(data, edges, rule_set=rule_list)
    return edges
    
    
def process_rules(data):
    rule_list = [i for i in range(1, 7)] # Example Rule 1 to Rule 6
    subsets = all_subsets(rule_list)

    combinations = {}

    for subset in subsets:
        combinations["+".join(subset)] = edge_construction(data, subset)

    return combinations
