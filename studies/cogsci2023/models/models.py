# from studies.cogsci2023.models.evc_coged import *
# from studies.cogsci2023.models.evc_congruency import *
# from studies.cogsci2023.models.evc_demand_selection import *
# from studies.cogsci2023.models.exp_learning import *
# from studies.cogsci2023.models.expected_value import *
# from studies.cogsci2023.models.prospect_theory import *
# from studies.cogsci2023.models.shepard_luce_choice import *
# from studies.cogsci2023.models.stevens_power_law import *
# from studies.cogsci2023.models.stroop_model import *
# from studies.cogsci2023.models.task_switching import *
# from studies.cogsci2023.models.tva import *
# from studies.cogsci2023.models.weber_fechner import *

from .evc_coged import *
from .evc_congruency import *
from .evc_demand_selection import *
from .exp_learning import *
from .expected_value import *
from .prospect_theory import *
from .shepard_luce_choice import *
from .stevens_power_law import *
from .stroop_model import *
from .task_switching import *
from .tva import *
from .weber_fechner import *

# model inventory

model_inventory = dict()
model_inventory["weber_fechner"] = (
    weber_fechner_metadata(),
    weber_fechner_data,
    weber_fechner_experiment,
)

model_inventory["stevens_power_law"] = (
    stevens_power_law_metadata(),
    stevens_power_law_data,
    stevens_power_law_experiment,
)

model_inventory["expected_value"] = (
    expected_value_theory_metadata(),
    expected_value_theory_data,
    expected_value_theory_experiment,
)

model_inventory["prospect_theory"] = (
    prospect_theory_metadata(),
    prospect_theory_data,
    prospect_theory_experiment,
)

model_inventory["exp_learning"] = (
    exp_learning_metadata(),
    exp_learning_data,
    exp_learning_experiment,
)

model_inventory["stroop_model"] = (
    stroop_model_metadata(),
    stroop_model_data,
    stroop_model_experiment,
)

model_inventory["task_switching"] = (
    task_switching_metadata(),
    task_switching_data,
    task_switching_experiment,
)

model_inventory["evc_coged"] = (
    evc_coged_metadata(),
    evc_coged_data,
    evc_coged_experiment,
)

model_inventory["evc_demand_selection"] = (
    evc_demand_metadata(),
    evc_demand_data,
    evc_demand_experiment,
)

model_inventory["evc_congruency"] = (
    evc_congruency_metadata(),
    evc_congruency_data,
    evc_congruency_experiment,
)

model_inventory["shepard_luce_choice"] = (
    shepard_luce_metadata(),
    shepard_luce_data,
    shepard_luce_experiment,
)

model_inventory["tva"] = (tva_metadata(), tva_data, tva_experiment)

plot_inventory = dict()
plot_inventory["weber_fechner"] = (plot_weber_fechner, "Weber-Fechner Law")
plot_inventory["stevens_power_law"] = (plot_stevens_power_law, "Stevens' Power Law")
plot_inventory["expected_value"] = (plot_expected_value, "Expected Utility Theory")
plot_inventory["prospect_theory"] = (plot_prospect_theory, "Prospect Theory")
plot_inventory["exp_learning"] = (plot_exp_learning, "Exponential Learning")
plot_inventory["stroop_model"] = (plot_stroop_model, "Stroop Model")
plot_inventory["task_switching"] = (plot_task_switching, "Task Switching Model")
plot_inventory["evc_coged"] = (plot_evc_coged, "EVC: Cognitive Effort Discounting")
plot_inventory["evc_demand_selection"] = (plot_evc_demand, "EVC: Demand Selection")
plot_inventory["evc_congruency"] = (plot_evc_congruency, "EVC: Distractor Effect")
plot_inventory["shepard_luce_choice"] = (plot_shepard_luce, "Shepard-Luce Choice Ratio")
plot_inventory["tva"] = (plot_tva, "Visual Attention Model")

param_dict = {
    'weber_fechner': 1,
    'stevens_power_law': 2,
    'expected_value': 2,
    'prospect_theory': 7,
    'exp_learning': 2,
    'stroop_model': 1,
    'task_switching': 7,
    'evc_coged': 2,
    'evc_demand_selection': 3,
    'evc_congruency': 2,
    'shepard_luce_choice': 2,
    'tva': 2
}
