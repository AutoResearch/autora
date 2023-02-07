import pickle

import pandas as pd

from autora.skl.bms import BMSRegressor
from autora.skl.darts import DARTSRegressor
from studies.bms.models.context_free.weber import weber

# get domains
weber_data = []
x1_domain = range(1, 10)
x2_domain = range(1, 10)

# make data
for x1 in x1_domain:
    for x2 in x2_domain:
        if x2 >= x1:
            weber_data.append([x1, x2, weber(x1, x2)])

weber_data = pd.DataFrame(weber_data)
print(weber_data.shape)
bms = BMSRegressor(epochs=500)
darts = DARTSRegressor(max_epochs=100)

X = weber_data.iloc[:, :2]
y = weber_data.iloc[:, -1:]

bms.fit(X, y)
darts.fit(X, y)

# save and load pickle file
file_name = "data/data.pickle"

with open(file_name, "wb") as f:
    # simulation configuration
    configuration = dict()
    configuration["domain"] = x1_domain, x2_domain
    configuration["data"] = X, y
    object_list = [configuration]

    pickle.dump(object_list, f)

if __name__ == "__main__":
    ...
