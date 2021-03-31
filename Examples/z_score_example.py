from MLlib.utils.misc_utils import read_data
from MLlib.models import z_score

x, y = read_data("datasets/z_score_dataset.txt")

z_score.get_outlier(y[0], threshold_value=3)
# threshold_value as per user's choice
