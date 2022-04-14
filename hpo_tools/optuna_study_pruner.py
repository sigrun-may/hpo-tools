# Copyright (c) 2022 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Optuna study pruner."""

import math
from itertools import combinations

import optuna


epsilon = 1 / 1000


def study_patience_pruner(trial, epsilon, warm_up_steps, patience):
    """TODO: add docstring."""
    # pruning of complete study
    if trial.number >= warm_up_steps:
        evaluation_metrics_of_completed_trials = []
        for _trial in trial.study.trials:
            if _trial.state == optuna.trial.TrialState.COMPLETE:
                evaluation_metrics_of_completed_trials.append(_trial.value)

        if len(evaluation_metrics_of_completed_trials) > patience:
            evaluation_metrics_within_patience = evaluation_metrics_of_completed_trials[-patience:]
            evaluation_metrics_before_patience = evaluation_metrics_of_completed_trials[:patience]

            best_value_within_patience = max(evaluation_metrics_within_patience)
            best_value_before_patience = max(evaluation_metrics_before_patience)

            # was best value of study before the number_of_similar_best_values?
            if best_value_before_patience + epsilon > best_value_within_patience:
                trial.study.stop()
                raise optuna.TrialPruned()


def study_no_improvement_pruner(trial, epsilon, warm_up_steps, number_of_similar_best_values, patience_for_unpruned_trials=None, threshold=None):
    """TODO: add docstring."""

    # stop study if no trial could be completed unitl patience_for_unpruned_trials
    if patience_for_unpruned_trials and trial.number >= patience_for_unpruned_trials:
        try:
            # check if any trial was completed and not pruned
            _ = trial.study.best_value
        except AttributeError:
            trial.study.stop()
            raise optuna.TrialPruned()

    # stop study if there is no improvement greater than epsilon
    if trial.number >= warm_up_steps:

        # pruning only after reaching a given threshold to prevent pruning based on results of naive classifier
        if threshold and (trial.study.StudyDirection.MAXIMIZE and trial.study.best_value < threshold) or (trial.study.StudyDirection.MINIMIZE and trial.study.best_value > threshold):
            return  # abort pruning as the given threshold is not reached yet

        # check if there is no improvement greater than epsilon
        evaluation_metrics_of_completed_trials = []
        for _trial in trial.study.trials:
            if _trial.state == optuna.trial.TrialState.COMPLETE:
                evaluation_metrics_of_completed_trials.append(_trial.value)

        if len(evaluation_metrics_of_completed_trials) >= number_of_similar_best_values:
            evaluation_metrics_of_completed_trials.sort(reverse=True)
            comb = list(
                combinations(
                    evaluation_metrics_of_completed_trials[:number_of_similar_best_values],
                    2,
                )
            )
            if all([math.isclose(a, b, abs_tol=epsilon) for a, b in comb]):
                print("best_value", trial.study.best_value)
                print(evaluation_metrics_of_completed_trials[:number_of_similar_best_values])
                print(evaluation_metrics_of_completed_trials)
                print(trial.number)
                trial.study.stop()
                raise optuna.TrialPruned()
