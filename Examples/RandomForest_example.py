from MLlib.models import RandomForestClassifier
from MLlib.utils.misc_utils import RFread_data


A, head = RFread_data('datasets/RandomForest.txt')

RForest_model = RandomForestClassifier()

RForest_model.predict(A, head, n_estimators=100)
