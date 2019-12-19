"""
Contains abstract functionality for learning locally linear sparse model.
"""
from __future__ import print_function
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestRegressor

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)
    def getRmse(self, actual_preds, target_preds, weights):
        rmse_val = 0
        for x in range(len(actual_preds)):
            rmse_val += weights[x]*(float(target_preds[x])- actual_preds[x])*(float(target_preds[x]) - actual_preds[x])
        #print(rmse_val)
        rmse_val = float(rmse_val)/sum(weights)
        return rmse_val
    
    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   test_data,
                                   test_labels,
                                   test_data_distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
 #       sum_weights = sum(weights)
 #       for i, val in enumerate(weights):
 #           weights[i] = weights[i] / sum_weights
 #       print(weights)
        labels_column = neighborhood_labels[:, label]
        #print("labels_column " + str(labels_column))
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        random_forest_model = RandomForestRegressor(n_estimators = 10)
        random_forest_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        rf_2_model = RandomForestRegressor(n_estimators = 2)
        rf_2_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
#        prediction_score = easy_model.score(
#            neighborhood_data[:, used_features],
#            labels_column, sample_weight=weights)
        #local = []
        
        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        easy_model_preds = []
        random_forest_model_preds  = []
        rf_2_model_preds = []
        nb_pos =0
        pos =0
        neg =0
        bb_labels = []
        mean_pred_test = 0
        for i in range(len(neighborhood_data)):
            pred_1 = labels_column[i]
#            easy_model.predict(neighborhood_data[i, used_features].reshape(1, -1))
            nb_pos += labels_column[i]>0.5
            pos += weights[i]*(pred_1)
            neg += weights[i]*(1- pred_1)
            bb_labels.append(labels_column[i])
            easy_model_preds.append(easy_model.predict(neighborhood_data[i, used_features].reshape(1, -1)))
            random_forest_model_preds.append(random_forest_model.predict(neighborhood_data[i, used_features].reshape(1, -1)))
            rf_2_model_preds.append(rf_2_model.predict(neighborhood_data[i, used_features].reshape(1, -1)))
            mean_pred_test += weights[i]* labels_column[i]
           # print(str(i) + ' ' + str(weights[i]))
            #print(easy_model.predict(neighborhood_data[i, used_features].reshape(1, -1)))
        mean_pred_test = float(mean_pred_test)/sum(weights)
        #print("............" + str(mean_pred_test))
        prediction_score, naive_prediction_score, rf_prediction_score, rf_2_prediction_score = self.rmse_explanation_test_data(
                                   test_data, 
                                   test_labels, 
                                   test_data_distances, 
                                   easy_model,
                                   random_forest_model,
                                   rf_2_model,
                                   label,
                                   num_features,
                                   mean_pred_test,
                                   feature_selection='auto')
        #print(prediction_score)
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            #print(label) #-----------------------------------------------------------------------
            print('Right:', neighborhood_labels[0, label])
       # print("pos " + str(pos))
       # print("neg " + str(neg))
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, naive_prediction_score, rf_prediction_score,rf_2_prediction_score, local_pred, bb_labels,pos/(pos + neg),weights)
                
    def rmse_explanation_test_data(self, 
                                   test_data, 
                                   test_labels, 
                                   test_data_distances, 
                                   explanation_model,
                                   random_forest_model,
                                   rf_2_model,
                                   label,
                                   num_features,
                                   mean_pred_test,
                                   feature_selection='auto'):
        #mean_pred_test is the weighted mean of training data. This will be predicted if we use naive classifier
        test_data_weights = self.kernel_fn(test_data_distances)
        test_target = test_labels[:, label]
        used_features = self.feature_selection(test_data,
                                               test_target,
                                               test_data_weights,
                                               num_features,
                                               feature_selection)
        test_predictions = [] #Linear model explanations for test data
        test_naive_predictions = []
        test_rf_predictions = []
        test_rf_2_predictions = []
        for i in range(len(test_data)):
            test_predictions.append(explanation_model.predict(test_data[i, used_features].reshape(1, -1)))
            test_naive_predictions.append(mean_pred_test)
            test_rf_predictions.append(random_forest_model.predict(test_data[i, used_features].reshape(1, -1)))
            test_rf_2_predictions.append(rf_2_model.predict(test_data[i, used_features].reshape(1, -1)))
        rmse_test_data_naive = self.getRmse(test_target, test_naive_predictions,test_data_weights)
        rmse_test_data = self.getRmse(test_target, test_predictions,test_data_weights)
        rmse_test_data_rf = self.getRmse(test_target, test_rf_predictions,test_data_weights)
        rmse_test_data_rf_2 = self.getRmse(test_target, test_rf_2_predictions,test_data_weights)

        return rmse_test_data, rmse_test_data_naive,rmse_test_data_rf,rmse_test_data_rf_2