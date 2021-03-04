def class_counts(rows, label):
    """
    find the frequency of items for each class in a dataset.

    PARAMETERS
    ==========

    rows: list
        A list of lists to store the rows whose
        predictions is to be determined.

    label: integer
        The index of the last column

    RETURNS
    =======

    counts
        A dictionary of predictions
    """
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        lbl = row[label]
        if lbl not in counts:
            counts[lbl] = 0
        counts[lbl] += 1
    return counts


def is_numeric(value):
    """
    Test if a value is numeric.

    PARAMETERS
    ==========

    value
        The value whose datatype is to be determined.

    RETURNS
    =======

    Boolean true if the value is numeric, else boolean false.
    """

    return isinstance(value, int) or isinstance(value, float)


class Question:
    """
    A Question is used to partition a dataset.

    This class just records a 'column number' and a
    'column value'. The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.

    ATTRIBUTES
    ==========

    val: None
        Initially None type, stores the value
        of the row corresponding to 'self.column'
        column.

    CONSTRUCTOR
    ===========

    __init__ :
        initialize column, value and head.

    METHODS
    =======

    match(example):
        To compare the feature value in the example
        to the feature value in the question.

    __repr__ :
        Helper method to print the question in
        a readable format.
    """

    val = None

    def __init__(self, column, value, head):
        self.column = column
        self.value = value
        self.head = head

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        self.val = example[self.column]
        if is_numeric(self.val):
            return self.val >= self.value
        return self.val == self.head[self.column]

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.val):
            condition = ">="
            return "Is %s %s %s?" % (
                self.val, condition, self.value)
        return "Is %s %s %s?" % (
                self.val, condition, self.head[self.column])


def partition(rows, question):
    """
    Partitions a dataset.

    For each row in the dataset,
    check if it matches the question.
    If so, add it to 'true rows',
    otherwise, add it to 'false rows'.

    PARAMETERS
    ==========

    rows: list
        A list of lists to store the rows
        of the dataset to be partitioned.

    question: object
        Object of the class Question.

    RETURNS
    =======

    true_rows
        A list of lists that stores the rows
        for which the split question evaluates
        to true.

    false_rows
        A list of lists that stores the rows
        for which the split question evaluates
        to false.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows, length):
    """
    Calculate the Gini Impurity for a list of rows.

    PARAMETERS
    ==========

    rows: list
        A list of lists to store the rows of the
        dataset.

    length: integer
        To store the number of rows in the dataset
        to calculate the probability.

    RETURNS
    =======

    impurity
        The gini impurity value for the corresponding
        set of rows.
    """

    label = len(rows[0])-1

    # Get the dictionary of the predictions
    counts = class_counts(rows, label)
    impurity = 1
    for lbl in counts:
        # Store the probability of the predictions
        prob_of_lbl = counts[lbl] / float(length)

        # Update the gini impurity
        impurity -= prob_of_lbl**2
    return impurity


def find_best_split(rows, head):
    """
    Find the best question to ask by iterating over
    every feature / value and
    calculating the information gain.

    PARAMETERS
    ==========

    rows: list
        A list of lists for which the best split
        is to be determined.

    head: list
        A list to store the headings of the
        columns of the rows of the dataset.

    RETURNS
    =======

    best_gini
        The least gini impurity corresponding to
        the best split.

    best_question
        The best question to split the dataset.
    """

    best_gini = 100  # keep track of the best gini impurity
    best_question = None  # keep train of the feature / value that produced it
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        # unique values in the column
        values = list(set([row[col] for row in rows]))

        for val in values:  # for each value

            question = Question(col, val, head)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # finding gini impurity for true_rows
            ginitrue = gini(true_rows, len(rows))

            # finding gini impurity for false_rows
            ginifalse = gini(false_rows, len(rows))

            # calculating weighted_gini as per formula
            mid = len(true_rows)*ginitrue + len(false_rows)*ginifalse
            weightedgini = mid/len(rows)

            # Condition to find the best split
            if weightedgini <= best_gini:
                best_gini, best_question = weightedgini, question

    # Update best gini to 0
    # if we have reached the leaf node
    if best_gini == 100:
        best_gini = 0

    return best_gini, best_question
