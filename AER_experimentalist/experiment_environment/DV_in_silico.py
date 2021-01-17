from AER_experimentalist.experiment_environment.variable import Variable

class DV_In_Silico(Variable):

    _variable_label = "DV"
    _name = "dependent variable"
    _units = "activation"
    _priority = 0
    _value_range = (0, 1)
    _value = 0
    _participant = None

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):

        super(DV_In_Silico, self).__init__(*args, **kwargs)

    def assign_participant(self, participant):
        self._participant = participant

    # Waits until specified time has passed relative to reference time
    def measure(self):
        measurement = self._participant.get_value(self._name)
        self.set_value(measurement)

    # Get whether this dependent variable is treated as covariate.
    def __is_covariate__(self):
        return self._is_covariate

    # Set whether this dependent variable is treated as covariate.
    def __set_covariate__(self, is_covariate):
        self._is_covariate = is_covariate