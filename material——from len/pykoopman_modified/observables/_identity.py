"""
Linear observables
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class Identity(BaseObservables):
    """
    A dummy observables class that simply returns its input.
    """

    def __init__(self):
        """
        Initialize the Identity class.

        This constructor initializes the Identity class which simply returns its input
        when transformed.
        """
        super().__init__()
        self.include_state = True

    def fit(self, x, y=None):
        """
        Fit the model to the provided measurement data.

        Args:
            x (array-like): The measurement data to be fit. It must have a shape of
                (n_samples, n_input_features).
            y (None): This parameter is retained for sklearn compatibility.

        Returns:
            self: Returns a fit instance of the class `pykoopman.observables.Identity`.

        Note:
            only identity mapping is supported for list of arb trajectories
        """
        x = validate_input(x)
        if not isinstance(x, list):
            self.n_input_features_ = self.n_output_features_ = x.shape[1]
            self.measurement_matrix_ = np.eye(x.shape[1]).T
        else:
            self.n_input_features_ = self.n_output_features_ = x[0].shape[1]
            self.measurement_matrix_ = np.eye(x[0].shape[1]).T

        self.n_consumed_samples = 0

        return self

    def transform(self, x):
        """
        Apply Identity transformation to data.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Measurement data to be transformed.

        Returns
        -------
        y: array-like, shape (n_samples, n_input_features)
            Transformed data (same as x in this case).
        """
        # TODO validate input
        check_is_fitted(self, "n_input_features_")
        return x

    def inverse(self, y):
        """
        Invert the transformation.

        This function satisfies
        :code:`self.inverse(self.transform(x)) == x`

        Parameters
        ----------
        y: array-like, shape (n_samples, n_output_features)
            Data to which to apply the inverse.
            Must have the same number of features as the transformed data

        Returns
        -------
        x: array-like, shape (n_samples, n_input_features)
            Output of inverse map applied to y.
            In this case, x is identical to y.
        """
        # TODO: validate input
        check_is_fitted(self, "n_input_features_")
        return y

    def get_feature_names(self, input_features=None):
        """
        Get the names of the output features.

        Parameters
        ----------
        input_features: list of string, length n_input_features,\
         optional (default None)
            String names for input features, if available. By default,
            the names "x0", "x1", ... ,"xn_input_features" are used.

        Returns
        -------
        output_feature_names: list of string, length n_ouput_features
            Output feature names.
        """
        check_is_fitted(self, "n_input_features_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features_)]
        else:
            if len(input_features) != self.n_input_features_:
                raise ValueError(
                    "input_features must have n_input_features_ "
                    f"({self.n_input_features_}) elements"
                )
        return input_features
