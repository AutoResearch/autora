from AER_experimentalist.experiment_environment.variable import Variable

class IV_In_Silico(Variable):

    _variable_label = "IV"
    _name = "independent variable"
    _units = "activation"
    _priority = 0
    _value_range = (0, 1)
    _value = 0
    _participant = None

    # Initializes Industrial Analog Out 2.0 device.
    def __init__(self, *args, **kwargs):

        super(IV_In_Silico, self).__init__(*args, **kwargs)

    def assign_participant(self, participant):
        self._participant = participant

    # Waits until specified time has passed relative to reference time
    def manipulate(self):
        self._participant.set_value(self._name, self.get_value())

    # Set whether this dependent variable is treated as covariate.
    def set_covariate(self, is_covariate):
        self._is_covariate = is_covariate
