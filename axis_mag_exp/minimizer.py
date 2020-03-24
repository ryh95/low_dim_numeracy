import time
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import f1_score, mean_squared_error
from keras import backend as K
from abc import ABC, abstractmethod

class Minimizer(ABC):

    def __init__(self,base_workspace, optimize_types, mini_func):
        self.base_workspace = base_workspace
        self.mini_func = mini_func
        self.optimize_types = optimize_types

    @abstractmethod
    def objective(self,feasible_point):
        pass

    def minimize(self,space,**min_args):

        return self.mini_func(self.objective, space, **min_args)

class MagnitudeAxisMinimizer(Minimizer):

    def objective(self,feasible_point):
        optimize_workspace = {type: type_values for type, type_values in zip(self.optimize_types, feasible_point)}

        X_train = self.base_workspace['X_train']
        y_train = self.base_workspace['y_train']
        X_val = self.base_workspace['X_val']
        y_val = self.base_workspace['y_val']

        if hasattr(self,'model_fixed_params'):
            model = self.model(**optimize_workspace,**self.model_fixed_params)
        else:
            model = self.model(**optimize_workspace)

        try:
            if isinstance(model,KerasRegressor):
                K.clear_session()
                model.fit(X_train,y_train,verbose=0)
            else:
                model.fit(X_train,y_train)
            y_pred_val = model.predict(X_val)
            error = mean_squared_error(y_val, y_pred_val)
        except np.linalg.LinAlgError:
            error = 1e+30
        return error