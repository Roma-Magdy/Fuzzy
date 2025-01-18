# import dependencies
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt


def five_number_summary(data):
    # Calculate the five-number summary
    min_val = np.min(data)
    Q1 = np.percentile(data, 25)
    median = np.median(data)
    Q3 = np.percentile(data, 75)
    max_val = np.max(data)
    iqr = Q3 - Q1
    return min_val, Q1, median, Q3, max_val, iqr


# uncomment the following functions to make a five-level membership function
"""
def smallest(x, Q1, iqr):
    if x <= (Q1-3*iqr):
        return 1
    elif (Q1-(3*iqr)) < x <= (Q1-1.5*iqr):
        return (Q1-(x+1.5*iqr))/(1.5*iqr)
    elif x > Q1-1.5*iqr:
        return 0
    else:
        return 0        
"""
"""  
def largest (x, Q3 ,iqr):
    if x <= (Q3+1.5*iqr):
        return 0
    elif Q3< x <= (Q3+(1.5*iqr)):
        return (x-(Q3+1.5*iqr))/(1.5*iqr)
    elif x> Q3+3*iqr:
        return 1
    else:
        return 0
"""


# define membership function for the purpose of generating rules
def small(x, Q1, iqr):
    if x <= (Q1 - 3 * iqr):
        return 0
    elif (Q1 - (3 * iqr)) < x <= (Q1 - (1.5 * iqr)):
        return (x - (Q1 - 3 * iqr)) / (1.5 * iqr)
    elif (Q1 - (1.5 * iqr)) < x <= Q1:
        return (Q1 - x) / (1.5 * iqr)
    elif x >= Q1:
        return 0
    else:
        return 0


def medium(x, Q1, Q3, iqr):
    if x <= (Q1 - 1.5 * iqr):
        return 0
    elif (Q1 - (1.5 * iqr)) < x <= Q1:
        return (x - (Q1 - 1.5 * iqr)) / (1.5 * iqr)
    elif Q1 < x <= Q3:
        return 1
    elif Q3 < x <= (Q3 + (1.5 * iqr)):
        return ((Q3 + (1.5 * iqr)) - x) / (1.5 * iqr)
    elif x >= Q3 + (1.5 * iqr):
        return 0
    else:
        return 0


def large(x, Q3, iqr):
    if x <= Q3:
        return 0
    elif Q3 < x <= (Q3 + (1.5 * iqr)):
        return (x - Q3) / (1.5 * iqr)
    elif (Q3 + (1.5 * iqr)) < x <= (Q3 + (3 * iqr)):
        return ((Q3 + (3 * iqr)) - x) / (1.5 * iqr)
    elif x >= Q3 + (3 * iqr):
        return 0
    else:
        return 0


def membership(data):
    # copy feature data

    membership_matrix = data.copy()
    # add a column for each level, the maximum value, level with the maximum value
    # to make a five-level membership function remove the following comments

    # membership_matrix['Smallest'] = np.nan
    # membership_matrix['Largest'] = np.nan
    membership_matrix["Small"] = np.nan
    membership_matrix["Medium"] = np.nan
    membership_matrix["Large"] = np.nan
    membership_matrix["max_value"] = np.nan
    membership_matrix["max_region"] = ""
    # define the linguistic variables
    # uncomment the following lines to make a five-level membership function

    # linguistic_vars = ['Smallest', 'Small', 'Medium', 'Large', 'Largest']
    linguistic_vars = ["Small", "Medium", "Large"]
    # create a dictionary to store fuzzy sets for each linguistic variable
    fuzzy_sets = {linguistic_var: [] for linguistic_var in linguistic_vars}
    min_val, Q1, median, Q3, max_val, iqr = five_number_summary(data)
    # convert the dataframe to an array

    data = data.values
    for i, x in enumerate(data):
        # Calculate membership values for each linguistic term
        # uncomment the following lines to make a five-level membership function

        """
        smallest_val = smallest(x, Q1, iqr)
        fuzzy_sets['Smallest'].append(smallest_val)
        membership_matrix.loc[i,'smallest'] = smallest_val
        """
        """
        largest_val = largest(x, Q3, iqr)
        fuzzy_sets['Largest'].append(largest_val)
        membership_matrix.loc[i,'Largest'] = largest_val
        """
        small_val = small(x, Q1, iqr)
        fuzzy_sets["Small"].append(small_val)
        membership_matrix.loc[i, "Small"] = small_val

        medium_val = medium(x, Q1, Q3, iqr)
        fuzzy_sets["Medium"].append(medium_val)
        membership_matrix.loc[i, "Medium"] = medium_val

        large_val = large(x, Q3, iqr)
        fuzzy_sets["Large"].append(large_val)
        membership_matrix.loc[i, "Large"] = large_val
        # assign the maximum value
        # uncomment the following line to make a five-level membership function

        # max_value =  max(smallest_val,small_val,medium_val,large_val,largest_val)
        max_value = max(small_val, medium_val, large_val)
        membership_matrix.loc[i, "max_value"] = max_value
        # assign the region with maximum value

        if max_value == small_val:
            membership_matrix.loc[i, "max_region"] = "Small"
        elif max_value == medium_val:
            membership_matrix.loc[i, "max_region"] = "Medium"
        elif max_value == large_val:
            membership_matrix.loc[i, "max_region"] = "Large"
        # uncomment the following lines to make a five-level membership function

        # elif(max_value==smallest_val):
        #    membership_matrix.loc[i,'max_region'] = 'Smallest'
        # elif(max_value==largest_val):
        #    membership_matrix.loc[i,'max_region'] = 'Largest'
    # convert each fuzzy set into a pandas series
    fuzzy_sets = {key: pd.Series(value) for key, value in fuzzy_sets.items()}

    return fuzzy_sets, membership_matrix


def rule_matrix(data, features_name, output_name):
    rules_str = []
    degrees = []
    rules = pd.DataFrame()
    # loop through a limited range for testing uncomment the following line for full dataset
    # for i in range(len(data)):
    for i in range(50):
        rule = ""
        degree = 0
        for j, feature in enumerate(features_name):
            if feature not in rules:
                rules[feature] = ""
            fuzzy_rules, membership_matrix = membership(data[[feature]])
            # assign the maximum region and calculate the rule degree(by multiplying the value of each part of rule)
            if j == 0:
                rule = "IF " + feature + " is " + membership_matrix.loc[i, "max_region"]
                degree = membership_matrix.loc[i, "max_value"]
                rules.loc[i, feature] = membership_matrix.loc[i, "max_region"]
            else:
                res_rule = (
                    "AND " + feature + " is " + membership_matrix.loc[i, "max_region"]
                )
                rule = ", ".join([rule, res_rule])
                rules.loc[i, feature] = membership_matrix.loc[i, "max_region"]
                degree = degree * membership_matrix.loc[i, "max_value"]
        fuzzy_rules, membership_matrix = membership(data[[output_name]])
        res_rule = (
            "THEN " + output_name + " is " + membership_matrix.loc[i, "max_region"]
        )
        rule = ", ".join([rule, res_rule])
        if output_name not in rules:
            rules[output_name] = ""
        rules.loc[i, output_name] = membership_matrix.loc[i, "max_region"]
        degree = degree * membership_matrix.loc[i, "max_value"]
        rules_str.append(rule)
        degrees.append(degree)
    rules["Rule"] = rules_str
    rules["Degree"] = degrees
    return rules


def fuzzy_rule_generator(data, features_name, output_name):
    antecedents = []
    rules_list = []
    # generate rules, sort them by degree, drop duplicated rules and keep the first

    rules = rule_matrix(data, features_name, output_name)
    rules = rules.sort_values(by="Degree", ascending=False)
    rules = rules.drop_duplicates(subset="Rule", keep="first")
    rules.reset_index(drop=True, inplace=True)

    for feature in features_name:
        min_val, _, _, _, max_val, _ = five_number_summary(data[[feature]])
        antecedent = ctrl.Antecedent(np.arange(min_val, max_val, 1), feature)
        # uncomment the following lines to make a five-level membership function

        # antecedent['smallest'] = fuzz.trimf(antecedent.universe,[min_val, min_val, min_val + (max_val - min_val) / 4])
        # antecedent['Highest'] = fuzz.trimf(antecedent.universe, [min_val + (max_val - min_val) * 3 / 4, max_val, max_val])
        antecedent["Small"] = fuzz.trimf(
            antecedent.universe,
            [
                min_val,
                min_val + (max_val - min_val) / 4,
                min_val + (max_val - min_val) / 2,
            ],
        )
        antecedent["Medium"] = fuzz.trimf(
            antecedent.universe,
            [
                min_val + (max_val - min_val) / 4,
                min_val + (max_val - min_val) / 2,
                min_val + (max_val - min_val) * 3 / 4,
            ],
        )
        antecedent["Large"] = fuzz.trimf(
            antecedent.universe,
            [
                min_val + (max_val - min_val) / 2,
                min_val + (max_val - min_val) * 3 / 4,
                max_val,
            ],
        )
        antecedents.append(antecedent)

    min_val, _, _, _, max_val, _ = five_number_summary(data[[output_name]])
    consequent = ctrl.Consequent(np.arange(min_val, max_val, 1), output_name)

    consequent["Small"] = fuzz.trimf(
        consequent.universe,
        [min_val, min_val + (max_val - min_val) / 4, min_val + (max_val - min_val) / 2],
    )
    consequent["Medium"] = fuzz.trimf(
        consequent.universe,
        [
            min_val + (max_val - min_val) / 4,
            min_val + (max_val - min_val) / 2,
            min_val + (max_val - min_val) * 3 / 4,
        ],
    )
    consequent["Large"] = fuzz.trimf(
        consequent.universe,
        [
            min_val + (max_val - min_val) / 2,
            min_val + (max_val - min_val) * 3 / 4,
            max_val,
        ],
    )
    # visualize the antecedents and consequent & print the rules
    print(rules.head(10))
    for antecedent in antecedents:
        antecedent.view()
        plt.show()
    consequent.view()
    plt.show()
    # create antecedents expression (featuer1[max_region] & featuer2[max_region]...so on) to create rule objects

    for i in range(len(rules)):
        rule_expression = None
        for j, antecedent in enumerate(antecedents):
            term = antecedent[rules.loc[i, antecedent.label]]
            if rule_expression is None:
                rule_expression = term
            else:
                rule_expression &= term
        rule = ctrl.Rule(rule_expression, consequent[rules.loc[i, consequent.label]])
        rules_list.append(rule)

    return antecedents, consequent, rules_list


def fuzzy_simulator(antecedents, consequent, rules_list):
    sprinkler_system = ctrl.ControlSystem(rules_list)
    sprinkler_simulator = ctrl.ControlSystemSimulation(sprinkler_system)
    # set input values for each antecedent
    for antecedent in antecedents:
        val = float(input("Enter the value of " + antecedent.label + ":"))
        sprinkler_simulator.input[antecedent.label] = val
    # compute the output based on the inputs
    sprinkler_simulator.compute()
    output_val = sprinkler_simulator.output[consequent.label]
    print(consequent.label, " value: ", output_val)
    # visualize what we get
    for antecedent in antecedents:
        antecedent.view(sprinkler_simulator)
        plt.show()
    consequent.view(sprinkler_simulator)
    plt.show()


data = pd.read_csv("csv_files/cleaned_diabetes.csv")

antecedents, consequent, rules_list = fuzzy_rule_generator(
    data, ["Age", "BMI"], "Glucose"
)
fuzzy_simulator(antecedents, consequent, rules_list)
