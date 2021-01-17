from AER_experimentalist.experiment_environment.experiment import Experiment

class Experiment_In_Silico(Experiment):

    # Initialization requires path to experiment file
    def __init__(self, path=None, main_directory = "", participant=None):
        super(Experiment_In_Silico, self).__init__(path, main_directory)

        self._ITI = 0.0  # inter trial interval in seconds
        self._measurement_onset_asynchrony = 0.0

        self.participant=participant

    def init_experiment(self):
        super(Experiment_In_Silico, self).init_experiment()

        for independent_variable in self.IVs:
            independent_variable.assign_participant(self.participant)

        for dependent_variable in self.DVs:
            dependent_variable.assign_participant(self.participant)


    # Delay measurement
    def delay_measurement(self):
        self.participant.execute()
        super(Experiment_In_Silico, self).delay_measurement()