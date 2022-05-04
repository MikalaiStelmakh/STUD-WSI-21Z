import numpy as np
import pandas as pd
import argparse


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

    parser = argparse.ArgumentParser(
        description="Naive Bayes Classifier"
    )
    parser.add_argument("file", type=str,
                        help="Path to dataset file.")
    parser.add_argument("--test_size", type=float, default=0.5,
                        help="Ration of test samples compared to \
                              train samples (default: 0.2).")
    parser.add_argument("--shuffle", action='store_true',
                        help="Shuffle dataset before classification \
                             (default: False).")
    args = parser.parse_args()

    data: pd.DataFrame = pd.read_csv(args.file, delimiter="\t", header=None, names=[
                      'area', 'perimeter', 'compactness', 'lengthOfKernel',
                      'widthOfKernel', 'asymmetryCoefficient',
                      'lengthOfKernelGroove', 'seedType'])
    # data = data.sample(frac=1, random_state=120)
    train, test = train_test_split(
        data, test_size=args.test_size, shuffle=args.shuffle
    )
    X_train = train.iloc[:, :-1].values
    Y_train = train.iloc[:, -1].values

    X_test = test.iloc[:, :-1].values
    Y_test = test.iloc[:, -1].values

    nb = NaiveBayes()
    nb.fit(X_train, Y_train)
    predictions = nb.predict(X_test)

    print("Confusion matrix\n", confusion_matrix(Y_test, predictions))
    print("Precision: ", list(precision_score(Y_test, predictions, average=None)))
    print("Accuracy: ", accuracy_score(Y_test, predictions))
    print("Recall: ", list(recall_score(Y_test, predictions, average=None)))
