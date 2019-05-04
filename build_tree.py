import regression_tree
import sys
import csv
import copy
import random as rd


def bootstrap(data, rows=None):
    rd.seed(1)
    buf = []
    train = []
    for rows in range(rows):
        buf = copy.deepcopy(data[rd.randint(0, len(data))])
        train.append(buf)
    return train


def main():
    if len(sys.argv) < 2:  # input file name should be specified
        print("Please specify input csv file name")
        return

    csv_file_name = sys.argv[1]

    data = []
    col_names = []
    line = 0
    with open(csv_file_name, encoding="Latin-1") as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            information = []
            for attribute in row:
                try:
                    information += [float(attribute)]
                except ValueError:
                    information += [attribute]
            if line == 0:
                col_names = information
                line = 1
            else:
                data.append(information)

    train = bootstrap(data, rows=int(len(data)/10 * 7))
    test = []
    for i in data[1:]:
        if i not in train:
            test.append(i)

    print("Total number of records for training = ", len(train))
    # print(len(test))
    # print(col_names)
    # print(len(col_names))
    tree = regression_tree.build_tree(train, min_gain=0.001, min_samples=20)

    regression_tree.print_tree(tree, '', col_names)

    max_tree_depth = regression_tree.max_depth(tree)
    print("max number of questions=" + str(max_tree_depth))

    age = float(input("What is your age?"))
    height = float(input("What is your height(cm)?"))
    weight = float(input("What is your weight(kg)?"))

    command = input("Enter 1 to enter a target value to find the requirement\n"
                    "Enter 2 to enter attributes to find value\nEnter 3 to quit:")
    while command != '3':
        if command == '1':
            list1 = []
            list2 = ""
            regression_tree.find_route(tree, list1, list2, age, height, weight, col_names)
            value = float(input("What is your target value?"))
            # print(list1)
            requirement = regression_tree.target_route(value, list1)
            print(requirement)
        elif command == '2':
            inputs = [None] * (len(col_names) - 1)
            for i in range(len(col_names) - 1):
                inputs[i] = float(input("what is the score of " + col_names[i] + ":"))
            # print(regression_tree.classify(inputs, tree))
            target = regression_tree.calculate_value(regression_tree.classify(inputs, tree))
            print("Your target value is: ", target)
        command = input("Enter 1 to enter a target value to find the requirement\n"
                        "Enter 2 to enter attributes to find value\nEnter 3 to quit:")


if __name__ == "__main__":
    main()
