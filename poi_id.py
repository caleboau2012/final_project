#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Using my intuition to add bonus, restrictied stock deferred, total_stock_value, exercised_stock_options and restricted_stock
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'bonus', 'restricted_stock_deferred', 'deferred_income',\
 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock',\
  'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# checking for outliers
# for key, value in data_dict.iteritems():
# 	print key

# removing total. It's a spreadsheet quirk
data_dict.pop("TOTAL", 0);

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## feature scaling to the rescue
from sklearn import preprocessing
features_scaled = preprocessing.scale(features)

## feature selection time, blessing time.
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

percentile = SelectPercentile(f_classif, percentile=10).fit(features_scaled, labels)
print features, percentile.scores_
features_new = percentile.transform(features_scaled)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.grid_search import GridSearchCV
parameters = {
          'learning_rate': [0.5, 1.0],
          }

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

from sklearn.ensemble import AdaBoostClassifier
algo = AdaBoostClassifier()

clf = GridSearchCV(algo, parameters)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_new, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train);

from sklearn.metrics import accuracy_score, precision_score, recall_score
ans = clf.predict(features_test)
# print ans

print accuracy_score(labels_test, ans)
print precision_score(labels_test, ans)
print recall_score(labels_test, ans)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)