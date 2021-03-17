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

from lime.discretize import QuartileDiscretizer
from lime.discretize import DecileDiscretizer
from lime.discretize import EntropyDiscretizer
from lime.discretize import BaseDiscretizer
from lime.discretize import StatsDiscretizer
import lime.explanation as Explanation
import lime.lime_base as lime_base

import matplotlib.pyplot as plt


class PASTLE_Explanation():

    def __init__(self,
                 dataset,
                 labels,
                 base_explanation,  # LIME explanation object
                 pivots,
                 feature_importances,
                 test_instance,
                 true_pred,  # black-box prediction
                 feature_names,
                 mode = 'classification',
                 verbose=False):

        self.base_exp = base_explanation
        self.test_instance = test_instance
        self.dataset = dataset
        self.labels = labels
        self.feature_names = np.array(feature_names)
        import matplotlib.cm as cm
        self.colors = cm.hsv(np.linspace(0, 1, len(self.feature_names)))
        self.pivots = pivots
        self.true_pred = true_pred
        self.mode = mode

        self.exp_vector, _, _, _ = self.get_exp_vector(test_instance, base_explanation, pivots, verbose)
        self.features_order = np.array([c[0] for c in feature_importances])
        self.features_importance = np.array([c[1] for c in feature_importances])

    def get_exp_vector(self, test_instance, base_exp, pivots, verbose):
        num_pivots = int(len(pivots))
        exp_pivots = []
        weights_array = np.zeros(num_pivots)

        if self.mode=='regression':
            x = 1
        else:
            x = base_exp.available_labels()[0]

        for pair in base_exp.local_exp[x]:
            pivot_idx = pair[0]
            exp_pivots.append(pivot_idx)
            weights_array[pivot_idx] = pair[1]

        distance_values = self.base_exp.distance_values
        components = weights_array * distance_values
        vectors = []

        for i in exp_pivots:
            u_dir = pivots[i] - test_instance
            u_amp = np.sqrt(sum(u_dir ** 2))
            v_amp = np.abs(components[i])
            v = v_amp * (u_dir / u_amp)
            vectors.append(v)
        vectors = np.array(vectors)

        exp_vector_P = sum(vectors[weights_array[exp_pivots] > 0])
        exp_vector_N = sum(vectors[weights_array[exp_pivots] < 0])
        exp_vector = exp_vector_P - exp_vector_N
        if verbose:
            print("WEIGHTS: ", weights_array);
            print("DISTANCES: ", distance_values);
            print("PIVOT_VECTORS_AMPs_withsign: ", components);
            print("Explanation vector: ", exp_vector);

        return exp_vector, distance_values, weights_array, components

    def show_in_notebook(self, save=False, name=None, num_features=100):
        def drawArrow(A, B, color):
            ax.annotate('', xy=(A[0], A[1]),
                        xycoords='data',
                        xytext=(B[0], B[1]),
                        textcoords='data',
                        arrowprops=dict(arrowstyle='<|-',
                                        color=color,
                                        lw=3.5,
                                        ls='-')
                        )

        idx = self.features_order[:num_features]
        fi = self.features_importance[:num_features]

        print("Index & Importance: ", idx, " - ", fi)
        x_coord = fi.reshape((len(self.feature_names[idx]), 1))
        if self.mode == 'classification':
            if self.base_exp.available_labels()[0] == 0:
                x_coord *= -1

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # ax.scatter(x_coord, y_coord, s=100, c='b', alpha=0.5)
        ax.grid(False, which='both')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_alpha(0.2)
        ax.spines['bottom'].set_zorder(0)
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if np.min(x_coord) < 0:
            leftlim = np.min(x_coord) - 0.1 * np.max(np.abs(x_coord))
        else:
            leftlim = 0
        rightlim = np.max(x_coord) + 0.1 * np.max(np.abs(x_coord))
        ax.set_xlim(leftlim, rightlim)
        ax.set_ylim(-3, 3)
        ax.set_xticks([])
        ax.set_yticks([])

        for i in range(x_coord.shape[0]):
            drawArrow([x_coord[i], 0], [x_coord[i], np.sign(self.exp_vector[idx[i]])], color=self.colors[idx[i]])
            plt.scatter(x_coord[i], 0, color=self.colors[idx[i]], s=50)

        
        if self.mode == 'classification':       
            ax.axvspan(0, np.max(x_coord) + 0.1 * np.max(np.abs(x_coord)), facecolor='green', alpha=0.2, label='_nolegend_')

            if np.min(x_coord) < 0:
                ax.axvspan(np.min(x_coord) - 0.1 * np.max(np.abs(x_coord)), 0, facecolor='red', alpha=0.2,
                        label='_nolegend_')

            x = 0 if self.mode=='regression' else self.base_exp.class_names[self.base_exp.available_labels()[0]]
                                                                            
            ax.text(0.5 * rightlim, 2.8, x,
                    horizontalalignment='center', verticalalignment='top', fontsize=24, alpha=0.7)
        else:
            x = 'Predicted value = ' + str(self.true_pred)
            ax.text(0.5 * rightlim, 2.8, x,
                    horizontalalignment='center', verticalalignment='top', fontsize=24, alpha=0.7)
        leg = ax.legend([self.feature_names[c] + " = " + "{:.2f}".format(self.test_instance[c]) \
                         for c in idx], prop={'size': 20}, bbox_to_anchor=(1.95, 0.5), loc='center right', ncol=1)
        leg.get_frame().set_linewidth(0.0)

        if save:
            plt.savefig(name, bbox_inches='tight')

    def show_in_notebook2(self):
        def drawArrow(A, color):
            ax.annotate('', xy=(0, 0),
                        xycoords='data',
                        xytext=(A[0], A[1]),
                        textcoords='data',
                        arrowprops=dict(arrowstyle='<|-',
                                        color=color,
                                        lw=3.5,
                                        ls='-')
                        )

        idx = self.features_order
        fi = self.features_importance

        x_coord = fi[np.argsort(idx)].reshape((len(self.feature_names), 1))

        if self.base_exp.available_labels()[0] == 0:
            x_coord *= -1

        y_coord = np.array(self.exp_vector).reshape((len(self.feature_names), 1))
        # y_coord = np.divide(y_coord,(np.abs(np.max(dataset,axis=0)-np.min(dataset,axis=0))).reshape((8,1)))

        points = np.concatenate((x_coord, y_coord), axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        # ax.scatter(x_coord, y_coord, s=100, c='b', alpha=0.5)
        ax.grid(False, which='both')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

        cmap = plt.get_cmap('gist_rainbow')
        colors = cmap(np.linspace(0, 1, len(x_coord)))

        import matplotlib.cm as cm
        colors = cm.gist_rainbow(np.linspace(0, 1, len(x_coord)))

        for i in range(x_coord.shape[0]):
            drawArrow(points[i, :], color=colors[i])
            ax.scatter(x_coord[i], y_coord[i], s=100, color=colors[i])

        # ax.legend([feature_names[c] + ' = ' + str(test_instance[c]) for c in range(len(feature_names))],prop={'size': 14})
        ax.set_xlabel('feature_importance', size=14)
        ax.set_ylabel('direction', size=14)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.legend([self.feature_names[c] + ' = ' + str(self.test_instance[c]) for c in range(len(self.feature_names))],
                  prop={'size': 14}, bbox_to_anchor=(1.4, 1), loc='upper right', ncol=1)

    def move_along_directions(self, model, n_points=2000, direction='both', verbose=True,save=False,name=None):
        dataset = self.dataset
        test_instance = self.test_instance
        exp_vector = self.exp_vector

        datasetMin = np.min(dataset, axis=0)
        datasetMax = np.max(dataset, axis=0)

        dimensionality = dataset.shape[1]

        x = np.linspace(0, n_points, n_points)
        x = np.repeat(x, dimensionality).reshape((n_points, dimensionality))
        pts_supporting = test_instance + x * exp_vector * np.std(dataset, axis=0)
        pts_supporting = np.where(pts_supporting >= datasetMin, pts_supporting, datasetMin)
        pts_supporting = np.where(pts_supporting <= datasetMax, pts_supporting, datasetMax)
        if self.mode == 'classification':
            preds_supporting = model.predict_proba(pts_supporting)[:, self.base_exp.available_labels()[0]]
        else:
            preds_supporting = model.predict(pts_supporting)

        stop_point = len(pts_supporting)

        pts_opposing = test_instance - x * exp_vector * np.std(dataset, axis=0)
        pts_opposing = np.where(pts_opposing >= datasetMin, pts_opposing, datasetMin)
        pts_opposing = np.where(pts_opposing <= datasetMax, pts_opposing, datasetMax)

        if self.mode == 'classification':
            preds_opposing = model.predict_proba(pts_opposing)[:, self.base_exp.available_labels()[0]]
        else:
            preds_opposing = model.predict(pts_opposing)

        most_opposing = pts_opposing[np.argmin(preds_opposing)]
        stop_point = len(pts_opposing)

        if verbose is True:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            ax.tick_params(bottom=False, labelsize=12)
            if self.mode == 'classification':
                ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_xlabel('Points along directions',size=24)
            ax.set_ylabel('Model prediction',size=24)

            if direction == 'both' or direction == 'supporting':
                xx = range(stop_point)
                yy = preds_supporting[:stop_point]
                ax.plot(xx, yy, color='#238823')
            if direction == 'both' or direction == 'opposing':
                xx = range(stop_point)
                yy = preds_opposing[:stop_point]
                ax.plot(xx, yy, color='#E42531')
            plt.grid(True)
            if self.mode == 'classification':
                plt.legend(['Supporting direction', 'Opposing direction'],prop={'size': 24})
            else:
                plt.legend(['Positive direction', 'Negative direction'],prop={'size': 24})


            if save:
                plt.savefig(name, bbox_inches='tight')

        return pts_supporting, preds_supporting, pts_opposing, preds_opposing

    def show_scatter_plot(self, feature_1_id, feature_2_id, feature_1_range=None, feature_2_range=None):
        test_instance = self.test_instance
        exp_vector = self.exp_vector
        pivots = self.pivots
        dataset = self.dataset
        labels = self.labels
        feature_names = self.feature_names
        true_pred = self.true_pred
        pivot_names = ["Pivot_" + str(i) for i in range(len(pivots))]

        p = np.polyfit([test_instance[feature_1_id], test_instance[feature_1_id] - exp_vector[feature_1_id]],
                       [test_instance[feature_2_id], test_instance[feature_2_id] - exp_vector[feature_2_id]], 1)

        if exp_vector[feature_1_id] < 0:
            x1 = np.linspace(np.min(dataset[:, feature_1_id]), test_instance[feature_1_id], 2)
            x2 = np.linspace(test_instance[feature_1_id], np.max(dataset[:, feature_1_id]), 2)
        else:
            x1 = np.linspace(test_instance[feature_1_id], np.max(dataset[:, feature_1_id]), 2)
            x2 = np.linspace(np.min(dataset[:, feature_1_id]), test_instance[feature_1_id], 2)

        y1 = np.array([i * p[0] + p[1] for i in x1])
        y2 = np.array([i * p[0] + p[1] for i in x2])

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        plt.xlabel(feature_names[feature_1_id])
        plt.ylabel(feature_names[feature_2_id])

        if feature_1_range is not None:
            xlimsx = test_instance[feature_1_id] - feature_1_range
            xlimdx = test_instance[feature_1_id] + feature_1_range
            plt.xlim(xlimsx, xlimdx)
        if feature_2_range is not None:
            ylimsx = test_instance[feature_2_id] - feature_2_range
            ylimdx = test_instance[feature_2_id] + feature_2_range
            plt.ylim(ylimsx, ylimdx)

        # ax.scatter(dataset[:,feature_1_id],dataset[:,feature_2_id], label = 'dataset', zorder=0, alpha = 0.5, c = labels, cmap = matplotlib.colors.ListedColormap(['blue','red']))

        cdict = {0: 'grey', 1: 'red'}
        # cdict = {0: '#ecf3fd', 1: '#fbe6e5'}
        for g in np.unique(labels):
            ix = np.where(labels == g)
            ax.scatter(dataset[:, feature_1_id][ix], dataset[:, feature_2_id][ix], c=cdict[g], label=g, alpha=1,
                       zorder=0)

        ax.scatter(pivots[:, feature_1_id], pivots[:, feature_2_id], color='black', label='pivot')

        if (feature_1_range is None) and (feature_2_range is None):
            for i in range(len(pivot_names)):
                ax.text(pivots[i][feature_1_id], pivots[i][feature_2_id], pivot_names[i], fontsize=12)
        else:
            for i in range(len(pivot_names)):
                if ((pivots[i][feature_1_id] >= xlimsx) and
                        (pivots[i][feature_1_id] <= xlimdx) and
                        (pivots[i][feature_2_id] >= ylimsx) and
                        (pivots[i][feature_2_id] <= ylimdx)):
                    ax.text(pivots[i][feature_1_id], pivots[i][feature_2_id], pivot_names[i], fontsize=12)

        ax.scatter(test_instance[feature_1_id], test_instance[feature_2_id], marker='^', s=200, alpha=1, color='yellow',
                   zorder=3, label='target instance')
        ax.text(test_instance[feature_1_id], test_instance[feature_2_id] - 0.1 * test_instance[feature_2_id],
                'prediction: ' + "{:.2f}".format(true_pred), color='black', horizontalalignment='center', va='bottom',
                bbox=dict(facecolor='yellow', alpha=0.7))
        # ax.annotate("",
        #        xy=(test_instance[feature_1_id], test_instance[feature_2_id]), xycoords='data',
        #        xytext=(0.8, 0.8), textcoords='data',
        #        arrowprops=dict(arrowstyle="->",
        #                        connectionstyle="arc3"),
        #        )
        ax.plot(x1, y1, color='green', ls='dashdot', lw=2, zorder=2, label='right direction')
        ax.plot(x2, y2, color='red', ls='dashdot', lw=2, zorder=2, label='wrong direction')

        ax.legend()