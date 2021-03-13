"""
NOTE: This module has been developed by adapting LIME code,
        which can be found at the following link:
        https://github.com/marcotcr/lime

Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
from functools import partial
import json
import warnings

import numpy as np
import scipy as sp
import pandas as pd
import scipy
import sklearn
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.utils import check_random_state

import lime.explanation as explanation
import lime.lime_base as lime_base
from lime.discretize import QuartileDiscretizer, StatsDiscretizer, EntropyDiscretizer, DecileDiscretizer

from sklearn.cluster import KMeans


class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""

    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, discretized_feature_names=None,
                 feature_indexes=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
            feature_indexes: optional feature indexes used in the sparse case
        """
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_indexes = feature_indexes
        self.scaled_row = scaled_row
        if sp.sparse.issparse(scaled_row):
            self.all_categorical = False
        else:
            self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             show_table: if False, don't show table visualization.
             show_all: if True, show zero-weighted features in the table.
        """
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]

        if self.feature_indexes is not None:
            # Sparse case: only display the non-zero values and importances
            fnames = [self.exp_feature_names[i] for i in self.feature_indexes]
            fweights = [weights[i] for i in self.feature_indexes]
            if show_all:
                out_list = list(zip(fnames,
                                    self.feature_values,
                                    fweights))
            else:
                out_dict = dict(map(lambda x: (x[0], (x[1], x[2], x[3])),
                                    zip(self.feature_indexes,
                                        fnames,
                                        self.feature_values,
                                        fweights)))
                out_list = [out_dict.get(x[0], (str(x[0]), 0.0, 0.0)) for x in exp]
        else:
            out_list = list(zip(self.exp_feature_names,
                                self.feature_values,
                                weights))
            if not show_all:
                out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(out_list, ensure_ascii=False), label, div_name)
        return ret


class TabularExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 pivots=None,  # aggiunta solo per spambase
                 mode="classification",
                 training_labels=None,
                 cluster_model=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None,
                 proximity_function=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode

        """ N  E  W """
        self.cluster_model = cluster_model
        # self.cluster_assignments = np.array(cluster_model.predict(training_data))

        if isinstance(self.cluster_model, sklearn.pipeline.Pipeline):
            # self.pivots, self.pivot_names, self.pivot_classes = self.generate_pivots(cluster_model['scaling'].transform(training_data), self.cluster_assignments, cluster_model['clustering'].majority_classes)
            # self.pivots = pivots
            self.pivots = cluster_model['scaling'].inverse_transform(cluster_model['clustering'].pivots)
            self.pivot_names = ['Pivot_' + str(i) for i in range(len(cluster_model['clustering'].pivots))]
        else:
            self.pivots = cluster_model.pivots
            self.pivot_names = ['Pivot_' + str(i) for i in range(len(cluster_model.pivots))]

        """ N  E  W """
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        self.training_data_stats = training_data_stats
        self.proximity_function = proximity_function
        # Check and raise proper error in stats are supplied in non-descritized path
        if self.training_data_stats:
            self.validate_training_data_stats(self.training_data_stats)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(training_data):
            # Set the discretizer if training data stats are provided
            if self.training_data_stats:
                discretizer = StatsDiscretizer(training_data, self.categorical_features,
                                               self.feature_names, labels=training_labels,
                                               data_stats=self.training_data_stats)

            if discretizer == 'quartile':
                self.discretizer = QuartileDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels)
            elif discretizer == 'decile':
                self.discretizer = DecileDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels)
            elif discretizer == 'entropy':
                self.discretizer = EntropyDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))

            # Get the discretized_training_data when the stats are not provided
            if (self.training_data_stats is None):
                discretized_training_data = self.discretizer.discretize(
                    training_data)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names

        # Though set has no role to play if training data stats are provided
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if training_data_stats is None:
                if self.discretizer is not None:
                    column = discretized_training_data[:, feature]
                else:
                    column = training_data[:, feature]

                feature_count = collections.Counter(column)
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = training_data_stats["feature_values"][feature]
                frequencies = training_data_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    @staticmethod
    def validate_training_data_stats(training_data_stats):
        """
            Method to validate the structure of training data stats
        """
        stat_keys = list(training_data_stats.keys())
        valid_stat_keys = ["means", "mins", "maxs", "stds", "feature_values", "feature_frequencies"]
        missing_keys = list(set(valid_stat_keys) - set(stat_keys))
        if len(missing_keys) > 0:
            raise Exception("Missing keys in training_data_stats. Details: %s" % (missing_keys))

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_pivots=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         verbose=False):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # test_instance_cluster = self.cluster_model.predict(data_row.reshape(1, -1))[0]

        if verbose is True:
            print("[TabularExplainer] - explain_instance. L'istanza da spiegare: ", data_row)
            # print("[LIMETabularExplainer] - explain_instance. Cluster di appartenenza: ", test_instance_cluster)

        #        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
        #            # Preventative code: if sparse, convert to csr format if not in csr format already
        #            data_row = data_row.tocsr()

        data, inverse = self.__data_inverse(data_row, num_samples)

        #        if sp.sparse.issparse(data):
        #            # Note in sparse case we don't subtract mean since data would become dense
        #            scaled_data = data.multiply(self.scaler.scale_)
        #            # Multiplying with csr matrix can return a coo sparse matrix
        #            if not sp.sparse.isspmatrix_csr(scaled_data):
        #                scaled_data = scaled_data.tocsr()
        #        else:
        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

        distances = sklearn.metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric='euclidean'
        ).ravel()

        centers = (self.pivots - self.scaler.mean_) / self.scaler.scale_

        if verbose is True:
            print("[TabularExplainer] - explain_instance. Vicini creati e scalati. L'istanza originale scalata: ",
                  data[0])

        yss = predict_fn(inverse)

        if verbose is True:
            print("[TabularExplainer] - explain_instance. Probabilità prima istanza (modello black box): ",
                  str(yss[0]))

        test_instance_distances = \
        sp.spatial.distance.cdist(scaled_data[0].reshape(1, -1), centers, metric=distance_metric)[0]

        num_clusters = num_pivots if num_pivots <= len(centers) else len(centers)
        argcenters = np.argsort(test_instance_distances)
        # argcenters = argcenters[0:num_clusters]
        argcenters = np.sort(argcenters)

        cluster_names = ['Pivot_' + str(i) for i in argcenters]

        if verbose:
            print("[LIMETabularExplainer] - explain_instance. Cluster names: ", cluster_names)

        data_df = pd.DataFrame(scaled_data)
        centers = pd.DataFrame(centers[argcenters])

        centroid_distances = scipy.spatial.distance.cdist(data_df.iloc[:, :], centers.iloc[:, :],
                                                          metric=distance_metric)
        centroid_distances = self.proximity_function(centroid_distances)
        #        if sp.sparse.issparse(data_row):
        #            values = self.convert_and_round(data_row.data)
        #            feature_indexes = data_row.indices
        #        else:
        values = self.convert_and_round(data_row)
        feature_indexes = None

        if self.mode == 'regression':
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]               
            

        domain_mapper = TableDomainMapper(cluster_names,
                                          centroid_distances[0],
                                          centroid_distances[0],  # N E W
                                          categorical_features=self.categorical_features,
                                          discretized_feature_names=None,
                                          feature_indexes=None)

        domain_mapper_lime = TableDomainMapper(self.feature_names,
                                               data[0],
                                               scaled_data[0],
                                               categorical_features=self.categorical_features,
                                               discretized_feature_names=None,
                                               feature_indexes=None)

        castle_exp = explanation.Explanation(domain_mapper,
                                             mode=self.mode,
                                             class_names=self.class_names)
        castle_exp.scaled_data = centroid_distances

        castle_exp.distance_values = centroid_distances[0]

        lime_exp = explanation.Explanation(domain_mapper_lime,
                                           mode=self.mode,
                                           class_names=self.class_names)
        lime_exp.scaled_data = scaled_data



        if self.mode == 'classification':
            castle_exp.predict_proba = yss[0]
            lime_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                castle_exp.top_labels = list(labels)
                castle_exp.top_labels.reverse()
                lime_exp.top_labels = list(labels)
                lime_exp.top_labels.reverse()
                labels = castle_exp.top_labels
        else:
            castle_exp.predicted_value = predicted_value
            castle_exp.min_value = min_y
            castle_exp.max_value = max_y
            lime_exp.predicted_value = predicted_value
            lime_exp.min_value = min_y
            lime_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (castle_exp.intercept[label],
             castle_exp.local_exp[label],
             castle_exp.score, castle_exp.local_pred) = self.base.explain_instance_with_data(
                centroid_distances,  # N E W
                yss,
                distances,
                label,
                num_clusters,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

            (lime_exp.intercept[label],
             lime_exp.local_exp[label],
             lime_exp.score, lime_exp.local_pred) = self.base.explain_instance_with_data(
                scaled_data,
                yss,
                distances,
                label,
                num_features,
                feature_selection=self.feature_selection,
                model_regressor=model_regressor)
        
        if self.mode == "regression":
            lime_exp.intercept[1] = lime_exp.intercept[0]
            lime_exp.local_exp[1] = [x for x in lime_exp.local_exp[0]]
            lime_exp.local_exp[0] = [(i, -1 * j) for i, j in lime_exp.local_exp[1]]

            castle_exp.intercept[1] = castle_exp.intercept[0]
            castle_exp.local_exp[1] = [x for x in castle_exp.local_exp[0]]
            castle_exp.local_exp[0] = [(i, -1 * j) for i, j in castle_exp.local_exp[1]]           

        return castle_exp, lime_exp

    def generate_pivots(self, dataset, cluster_assignments, majority_classes):
        num_features = dataset.shape[1]
        num_pivots = num_features + 2
        clusters, counts = np.unique(cluster_assignments[cluster_assignments >= 0], return_counts=True)
        majority_classes_tomask = majority_classes[clusters]
        pivots_per_cluster = np.zeros(np.max(clusters) + 1).astype(int)
        n_classes = np.max(majority_classes_tomask) + 1

        print("CLUSTERS: ", clusters, " - COUNTS: ", counts)
        for i in range(int(n_classes)):
            mask = (majority_classes_tomask == i)
            print(mask, clusters[mask], counts[mask])
            pivots_per_cluster[clusters[mask]] = np.round(
                counts[mask] / np.sum(counts[mask]) * (num_pivots / (n_classes))).astype(int)
            print(pivots_per_cluster[clusters[mask]])
            print(pivots_per_cluster)
        print("PIVOTS_PER_CLUSTER: ", pivots_per_cluster)

        pivots = []
        names = []
        classes = []
        for i in range(len(pivots_per_cluster)):
            name = 'Cluster_' + str(i)
            if pivots_per_cluster[i] != 0:
                model = KMeans(n_clusters=pivots_per_cluster[i], init='k-means++', max_iter=300, n_init=10,
                               random_state=0).fit(dataset[cluster_assignments == i])
                pivots.append(model.pivots)
                names.append([name + '_P' + str(j) for j in range(pivots_per_cluster[i])])
                classes.append([majority_classes[i] for j in range(pivots_per_cluster[i])])

        flat_list = [item for sublist in pivots for item in sublist]
        pivots = np.array(flat_list)

        flat_list = [item for sublist in names for item in sublist]
        names = np.array(flat_list)

        flat_list = [item for sublist in classes for item in sublist]
        classes = np.array(flat_list)

        return pivots, names, classes

    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]
            data = self.random_state.normal(
                0, 1, num_samples * num_cols).reshape(
                num_samples, num_cols)
            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse