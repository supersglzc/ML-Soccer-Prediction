from math import log
import pickle


class decision_node:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb  # True branch
        self.fb = fb  # False branch


# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows, column, value):
    # Make a function that tells
    # if the split is on numeric value (>=/<)
    # or on nominal value (==/!=)
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value

    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return set1, set2


# Create counts of class labels for a given set
# (the last column of each row is the class attribute)
def uniquecounts(rows):
    results = {}
    for row in rows:
        # The result is the last column
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


# Probability that a randomly placed item will
# be in the wrong category
# Using the closed form here:
# https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
def giniimpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


# Entropy is the sum of p(x)log(p(x)) across all
# the different possible classes
def entropy(rows):
    results = uniquecounts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log(p, 2)
    return ent


def variance(rows):
    if len(rows) == 0: return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    var = sum([(d - mean) ** 2 for d in data]) / len(data)
    return var


def prediction(leaf_labels):
    total = 0
    result = {}
    for label, count in leaf_labels.items():
        total += count
        result[label] = count

    for label, val in result.items():
        result[label] = str(int(result[label]/total * 100))+"%"

    return result


def prediction2(leaf_labels):
    total = 0
    result = 0
    for label, count in leaf_labels.items():
        total += count
        result += count*float(label)
    return round(float(result/total), 2)


def classify(observation, tree):
    if tree.results is not None:
        return prediction(tree.results)
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


# Classify an observation with missing data
def mdclassify(observation, tree):
    if tree.results is not None:
        return prediction(tree.results)
    else:
        v = observation[tree.col]
        if v is None:
            tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
            t_count = sum(tr.values())
            f_count = sum(fr.values())
            tw = float(t_count) / (t_count + f_count)
            fw = float(f_count) / (t_count + f_count)
            result = {}
            for k, v in tr.items(): result[k] = v * tw
            for k, v in fr.items(): result[k] = v * fw
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return mdclassify(observation, branch)


def build_tree(rows, score=variance,
               min_gain=0, min_samples=0):
    if len(rows) == 0:
        return decision_node()
    current_score = score(rows)

    # Set up accumulator variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # Generate the list of different values in
        # this column
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1
        # Now try dividing the rows up for each value
        # in this column
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)

            # Information gain
            p = float(len(set1)) / len(rows)
            gain = current_score - p * score(set1) - (1 - p) * score(set2)
            if gain > best_gain and len(set1) > min_samples and len(set2) > min_samples and gain > min_gain:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # Create the sub branches
    if best_gain > 0:
        true_branch = build_tree(best_sets[0], score, min_gain, min_samples)
        false_branch = build_tree(best_sets[1], score, min_gain, min_samples)
        return decision_node(col=best_criteria[0], value=best_criteria[1],
                             tb=true_branch, fb=false_branch)
    else:
        return decision_node(results=uniquecounts(rows))


def max_depth(tree):
    if tree.results is not None:
        return 0
    else:
        # Compute the depth of each subtree
        t_depth = max_depth(tree.tb)
        f_depth = max_depth(tree.fb)

        # Use the larger one
        if t_depth > f_depth:
            return t_depth + 1
        else:
            return f_depth + 1


def print_tree(tree, current_branch, attributes=None, indent='', leaf=prediction2):
    # Is this a leaf node?
    if tree.results is not None:
        print(indent + current_branch + str(leaf(tree.results)))
    else:
        # Print the split question
        split_col = str(tree.col)
        if attributes is not None:
            split_col = attributes[tree.col]
        split_val = str(tree.value)
        if type(tree.value) == int or type(tree.value) == float:
            split_val = ">=" + str(tree.value)
        print(indent + current_branch + split_col + ': ' + split_val + '? ')

        # Print the branches
        indent = indent + '  '
        print_tree(tree.tb, 'T->', attributes, indent)
        print_tree(tree.fb, 'F->', attributes, indent)


def find_route(tree, big_list, small_list, age, height, weight, attributes=None, leaf=prediction2):
    if tree.results is not None:
        big_list.append(small_list + "|" + str(leaf(tree.results)))
        return
    else:

        column = attributes[tree.col]
        value = str(tree.value)
        if column == 'Age':
            if float(age) <= float(value):
                find_route(tree.fb, big_list, small_list, age, height, weight, attributes)
            else:
                find_route(tree.tb, big_list, small_list, age, height, weight, attributes)
        elif column == 'Height(CM)':
            if float(height) <= float(value):
                find_route(tree.fb, big_list, small_list, age, height, weight, attributes)
            else:
                find_route(tree.tb, big_list, small_list, age, height, weight, attributes)
        elif column == 'Weight(KG)':
            if float(weight) <= float(value):
                find_route(tree.fb, big_list, small_list, age, height, weight, attributes)
            else:
                find_route(tree.tb, big_list, small_list, age, height, weight, attributes)
        else:
            if type(tree.value) == int or type(tree.value) == float:
                value = str(tree.value)
            str1 = column + " " + value + " " + "T"
            str2 = column + " " + value + " " + "F"
            buffer1 = small_list + "|" + str1
            buffer2 = small_list + "|" + str2
            find_route(tree.tb, big_list, buffer1, age, height, weight, attributes)
            find_route(tree.fb, big_list, buffer2, age, height, weight, attributes)


def target_route(target_value, data):
    dict2 = {}
    difference = 100000000000000000000000
    target = []
    for i in data:
        buffer = i.split("|")
        value = buffer[len(buffer) - 1]
        if abs(float(value) - target_value) < difference:
            difference = abs(float(value) - target_value)
            target = buffer
    target.pop(0)
    # target.insert(3, 'age 60.0 F')
    # print(target)
    for j in range(len(target) - 1):
        buffer = target[j].split(" ")
        key = buffer[0]
        if key not in dict2.keys():
            dict2[key] = [buffer[1], buffer[2]]
        else:
            if dict2[key][len(dict2[key]) - 1] == buffer[2]:
                if buffer[2] == 'F':
                    if float(dict2[key][len(dict2[key]) - 2]) < float(buffer[1]):
                        continue
                    else:
                        dict2[key][len(dict2[key]) - 2] = buffer[1]
                else:
                    if float(dict2[key][len(dict2[key]) - 2]) < float(buffer[1]):
                        dict2[key][len(dict2[key]) - 2] = buffer[1]
                    else:
                        continue
            elif dict2[key][1] == buffer[2]:
                if buffer[2] == 'F':
                    if float(dict2[key][0]) < float(buffer[1]):
                        continue
                    else:
                        dict2[key][0] = buffer[1]
                else:
                    if float(dict2[key][0]) < float(buffer[1]):
                        dict2[key][0] = buffer[1]
                    else:
                        continue
            else:
                dict2[key].append(buffer[1])
                dict2[key].append(buffer[2])
    # print(dict2)
    for key in dict2.keys():
        if len(dict2[key]) == 2:
            if dict2[key][1] == 'T':
                dict2[key] = key + ">=" + dict2[key][0]
            else:
                dict2[key] = key + "<=" + dict2[key][0]
        elif len(dict2[key]) == 4:
            if float(dict2[key][0]) > float(dict2[key][2]):
                dict2[key] = dict2[key][2] + "<=" + key + "<=" + dict2[key][0]
            else:
                dict2[key] = dict2[key][0] + "<=" + key + "<=" + dict2[key][2]
    return dict2


def calculate_value(dic):
    total = 0
    for u, v in dic.items():
        buffer = v.split("%")
        total += float(u) * (float(buffer[0]) / 100)
    return round(total, 1)


def save_tree(tree,name):
    pkl=open(name,'wb')
    string=pickle.dumps(tree)
    pkl.write(string)
    pkl.close()

def load_tree(name):
    with open(name,'rb')as file:
        tree=pickle.loads(file.read())
    return tree

def save_colName(colname):
    pkl=open('col_name.pkl','wb')
    string=pickle.dumps(colname)
    pkl.write(string)
    pkl.close()

def load_colName(name):
    with open(name,'rb')as file:
        name=pickle.loads(file.read())
    return name
