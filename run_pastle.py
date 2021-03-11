import time
import argparse

from src import data_handlers, pivots_generators
from src.explainers import TabularExplainer

import sklearn as sklearn
import sklearn.ensemble
import scipy
import numpy as np
import json

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return np.round(m,3), np.round(m-h,3), np.round(m+h,3), np.round(h,3)

data_handlers = {
    'adult': data_handlers.AdultDataHandler('data/adult'),
    'banknote': data_handlers.BanknoteDataHandler('data/banknote'),
    'diabetes': data_handlers.DiabetesDataHandler('data/diabetes'),
    'digits': data_handlers.DigitsDataHandler('data/digits'),
    'magic': data_handlers.MagicDataHandler('data/magic'),
    'spambase': data_handlers.SpambaseDataHandler('data/spambase'),
    'titanic': data_handlers.TitanicDataHandler('data/titanic')
}

models = {
    'random_forest': sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
}

proximity_fns = {
    'p1': (lambda x: 1/(1+x)),
    'p2': (lambda x: np.exp(-x)),
    'p3': (lambda x: -x),
    'p4': (lambda x: 1-(x - np.min(x,axis=0))/(np.max(x,axis=0) - np.min(x,axis=0))),
    'p5': (lambda x: np.max(x,axis = 0) - x),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default='random_forest')
    parser.add_argument('--proximity_fn', type=str, default='p1')
    parser.add_argument('--distance_fn', type=str, default='euclidean')
    parser.add_argument('--pivots_generation', type=str, default='random')
    parser.add_argument('--test_size', type=int, default=50)
    parser.add_argument('--n_pivots', type=int, default=-1)
    parser.add_argument('--perturbation_perc_range', type=float, default=None)
    parser.add_argument('--lime_discretize_continuous', type=bool, default=False)
    parser.add_argument('--lime_sample_around_instance', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_features', type=int, default=-1)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--log_all_results', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    print('Processing...')

    results = {}

    data_handler = data_handlers[args.dataset]
    model = models[args.model]

    data_handler.split(args.test_size, args.seed)
    model.fit(data_handler.X_train, data_handler.y_train)
    results['model_accuracy'] = sklearn.metrics.accuracy_score(data_handler.y_test, model.predict(data_handler.X_test))

    proximity_fn = proximity_fns[args.proximity_fn]

    if args.n_pivots == -1:
        args.n_pivots = data_handler.X_train.shape[1] + 1

    if args.n_features == -1:
        args.n_features = data_handler.X_train.shape[1]

    if args.pivots_generation == 'random':
        pivots_generator = pivots_generators.Random(data_handler.X_train, args.n_pivots)
    elif args.pivots_generation == 'perturbation':
        pivots_generator = pivots_generators.Perturbation(data_handler.X_train, args.n_pivots,
                                                          args.perturbation_perc_range)
    elif args.pivots_generation == 'clustering':
        pivots_generator = pivots_generators.Clustering(data_handler.X_train, args.n_pivots)

    explainer = TabularExplainer(data_handler.X_train,
                                 cluster_model=pivots_generator,
                                 feature_names=data_handler.feature_names,
                                 class_names=data_handler.class_names,
                                 discretize_continuous=args.lime_discretize_continuous,
                                 sample_around_instance=args.lime_sample_around_instance,
                                 verbose=args.verbose,
                                 proximity_function=proximity_fn)

    res_lime = []
    res_pastle = []

    for test_instance in data_handler.X_test:
        exp_pastle, exp_lime = explainer.explain_instance(test_instance,
                                                          model.predict_proba,
                                                          top_labels=1,
                                                          num_pivots=args.n_pivots,
                                                          num_samples=args.n_samples,
                                                          verbose=args.verbose,
                                                          distance_metric=args.distance_fn)

        pastle_adjusted_r2 = 1 - (1 - exp_pastle.score) * (args.n_samples - 1) / (args.n_samples - args.n_pivots - 1)
        lime_adjusted_r2 = 1 - (1 - exp_lime.score) * (args.n_samples - 1) / (args.n_samples - args.n_features - 1)

        if args.verbose == True:
            print("LIME: ", lime_adjusted_r2, " | PASTLE: ", pastle_adjusted_r2)

        res_lime.append(lime_adjusted_r2)
        res_pastle.append(pastle_adjusted_r2)

    differences = np.array(res_pastle) - np.array(res_lime)
    CI_diff, CI_pastle, CI_lime = mean_confidence_interval(differences), mean_confidence_interval(
        res_pastle), mean_confidence_interval(res_lime)

    if args.log_all_results is True:
        results['res_lime'] = res_lime
        results['res_pastle'] = res_pastle
    results['res_differences_mean'] = CI_diff[0]
    results['res_differences_CI'] = CI_diff[-1]
    results['res_pastle_mean'] = CI_pastle[0]
    results['res_pastle_CI'] = CI_pastle[-1]
    results['res_lime_mean'] = CI_lime[0]
    results['res_lime_CI'] = CI_lime[-1]
    results['config'] = vars(args)

    print(json.dumps(results, indent = 4))
    if args.log_dir is not None:
        log_file_name = args.dataset + '_npivots=' + str(args.n_pivots) + '_model=' + args.model + \
                        '_proximityfn=' + args.proximity_fn + '_distancefn=' + args.distance_fn + \
                        '_pivots_generation=' + args.pivots_generation
        if args.pivots_generation == 'perturbation':
            log_file_name = log_file_name + '_percrange=' + args.perturbation_perc_range

        print("Storing in "+ args.log_dir + '/' + log_file_name + '.json')
        json.dump(results, open(args.log_dir + '/' + log_file_name + '.json', 'w'))

if __name__ == '__main__':
    main()