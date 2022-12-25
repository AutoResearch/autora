from models import model_inventory
from autora.cycle import Cycle
from autora.experimentalist.pipeline import Pipeline
from autora.experimentalist.pooler import grid_pool, poppernet_pool
from autora.experimentalist.sampler import nearest_values_sampler
from autora.skl.bms import BMSRegressor
from autora.skl.darts import DARTSRegressor
from autora.variable import Variable, VariableCollection

# meta parameters
ground_truth_resolution = 1000
samples_per_cycle = 3
bms_epochs = 10
repetitions = 1

# SELECT THEORIST
# options: BMS, DARTS
theorist  = "BMS"

# SELECT GROUND TRUTH MODEL
# options: weber_fechner, exp_learning, stroop_model
study = "weber_fechner"

# SELECT EXPERIMENTALIST
# options: grid_pool, poppernet_pool

# SET UP STUDY

# build theorist
if theorist == "BMS":
    bms_theorist = BMSRegressor(epochs=bms_epochs)
elif theorist == "DARTS":
    darts_theorist = DARTSRegressor(epochs=bms_epochs)
else:
    raise ValueError(f"Theorist {theorist} not implemented.")

# build experimentalist
if study not in model_inventory.keys():
    raise ValueError(f"Study {study} not found in model inventory.")
(metadata,
 filter,
 experiment) = model_inventory[study]



