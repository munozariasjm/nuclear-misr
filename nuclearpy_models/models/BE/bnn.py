import pymc3 as pm
import theano.tensor as T
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BayesianNeuralNetwork:
    def __init__(
        self,
        path_to_dataset,
        target_column,
        feature_columns,
        test_size=0.2,
        random_state=42,
    ):
        self.data = pd.read_csv(path_to_dataset)
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.neural_network = None
        self.trace = None
        self.inference = None
        self.approx = None
        self.n_hidden = 5

    def preprocess_data(self):
        features = [
            self.scaler.fit_transform(self.data[col].values.reshape(-1, 1))
            for col in self.feature_columns
        ]
        self.X = np.column_stack(features)
        self.y = self.scaler.fit_transform(
            self.data[self.target_column].values.reshape(-1, 1)
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def construct_nn(self, ann_input, ann_output):
        init_1 = np.random.randn(self.X.shape[1], self.n_hidden).astype(float)
        init_2 = np.random.randn(self.n_hidden, self.n_hidden).astype(float)
        init_out = np.random.randn(self.n_hidden).astype(float)

        with pm.Model() as neural_network:
            weights_in_1 = pm.Normal(
                "w_in_1",
                0,
                sigma=1,
                shape=(self.X.shape[1], self.n_hidden),
                testval=init_1,
            )
            weights_1_2 = pm.Normal(
                "w_1_2",
                0,
                sigma=1,
                shape=(self.n_hidden, self.n_hidden),
                testval=init_2,
            )
            weights_2_out = pm.Normal(
                "w_2_out", 0, sigma=1, shape=(self.n_hidden,), testval=init_out
            )

            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
            act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

            out = pm.Normal(
                "out", act_out, observed=ann_output, total_size=self.y_train.shape[0]
            )
        return neural_network

    def train(self, n=30000):
        self.preprocess_data()
        self.neural_network = self.construct_nn(self.X_train, self.y_train)
        with self.neural_network:
            self.inference = pm.ADVI()
            self.approx = pm.fit(n, method=self.inference)
            self.trace = self.approx.sample(draws=5000)

    def test(self, samples=500):
        nn_input = T.shared(self.X_test)
        nn_output = T.shared(self.y_test)
        neural_network_test = self.construct_nn(nn_input, nn_output)
        with neural_network_test:
            ppc = pm.sample_posterior_predictive(self.trace, samples=samples)

        pred = ppc["out"].mean(axis=0)
        pred_std = ppc["out"].std(axis=0)
        return pred, pred_std
