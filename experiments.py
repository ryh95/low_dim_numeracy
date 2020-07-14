
class BaseExperiments(object):

    def __init__(self, exp_name, exp_data):
        self.name = exp_name
        self.exp_data = exp_data
        self.res = {}

    def prepare_datasets(self):
        pass

    def show_benchmark_res(self):
        pass

    def show_model_res(self):
        pass

    def save_res(self):
        pass

    def run(self):
        pass