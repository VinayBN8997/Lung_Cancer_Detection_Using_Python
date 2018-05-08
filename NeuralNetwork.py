from sklearn.neural_network import MLPClassifier
from datafile import dataset,dataset_output
import pickle

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(40, 10), random_state=1,learning_rate_init=0.001,max_iter=10000)
clf.fit(dataset, dataset_output)

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))