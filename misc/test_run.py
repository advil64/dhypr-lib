from datasets.air import Air
from models.link_pred import LPModel
from datasets.wiki import Wiki


my_dataset = Air('link_prediction')
# model = LPModel(my_dataset.get(0), proximity=my_dataset.proximity)

my_dataset

h = model.encode()