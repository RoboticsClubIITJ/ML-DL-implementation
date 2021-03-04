from MLlib.models import DecisionTreeClassifier
from MLlib.utils.misc_utils import RFread_data


A, head = RFread_data('datasets/DecisionTree.txt')

DTree_model = DecisionTreeClassifier()

DTree_model.print_tree(A, head)
