import unittest

from aer.variable import ValueType, Variable, VariableCollection


class TestVariableCollection(unittest.TestCase):
    def test_variablecollection_with_no_variables(self):
        VariableCollection(independent_variables=[], dependent_variables=[])

    def test_variablecollection_with_one_iv_one_dv(self):
        VariableCollection(
            independent_variables=[
                Variable(
                    name="S1",
                    value_range=(0, 5),
                    units="intensity",
                    variable_label="Stimulus 1 Intensity",
                )
            ],
            dependent_variables=[
                Variable(
                    name="difference_detected",
                    value_range=(0, 1),
                    units="probability",
                    variable_label="P(difference detected)",
                    type=ValueType.SIGMOID,
                )
            ],
        )

    def test_variablecollection_with_two_ivs_one_dv(self):
        a = VariableCollection(
            independent_variables=[
                Variable(
                    name="S1",
                    value_range=(0, 5),
                    units="intensity",
                    variable_label="Stimulus 1 Intensity",
                ),
                Variable(
                    name="S2",
                    value_range=(0, 5),
                    units="intensity",
                    variable_label="Stimulus 2 Intensity",
                ),
            ],
            dependent_variables=[
                Variable(
                    name="difference_detected",
                    value_range=(0, 1),
                    units="probability",
                    variable_label="P(difference detected)",
                    type=ValueType.SIGMOID,
                )
            ],
        )
        print(a)

    def test_variablecollection_with_one_iv_one_dv_one_cv(self):
        a = VariableCollection(
            independent_variables=[
                Variable(
                    name="S1",
                    value_range=(0, 5),
                    units="intensity",
                    variable_label="Stimulus 1 Intensity",
                )
            ],
            dependent_variables=[
                Variable(
                    name="difference_detected",
                    value_range=(0, 1),
                    units="probability",
                    variable_label="P(difference detected)",
                    type=ValueType.SIGMOID,
                )
            ],
            covariates=[
                Variable(
                    name="S2",
                    value_range=(0, 5),
                    units="intensity",
                    variable_label="Stimulus 2 Intensity",
                )
            ],
        )
        print(a)
