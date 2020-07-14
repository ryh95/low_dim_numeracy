from sub_mag_exp.helper.minimizer import Minimizer


class PredictionMinimizer(Minimizer):

    def objective(self, feasible_point):

        workspace = self.prepare_workspace(feasible_point)

        model = self.prepare_models(workspace)

        # if 'val_data' in workspace:
        #     # init evaluate
        #     print('evaluate on val data')
        #     evaluate_acc, _ = model.evaluate(workspace['val_data'])
        #     print(evaluate_acc)

        self.train(workspace,model,workspace['train_data'])

        self.save_models(workspace,model)

        MdAE, MdAPE, AVGR = self.evaluator.evaluate(model)
        self.evaluator.update_best_model(model)

        return AVGR