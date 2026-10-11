# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-13

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-13

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""KNN Classification Class"""
class Knn_Classification:
    # Constructor
    def __init__(self, x_train, y_train):
        """ Constructs a KNN classification object.
            :param x_train: The standardized training attributes.
            :type x_train: numpy.ndarray
            :param y_train: The training labels.
            :type y_train: numpy.ndarray
            :return: None
        """
        # Data fields
        self._x_train = x_train
        self._y_train = y_train

    # Getters
    def get_x_train(self):
        """ Returns the training attributes.
            :return: The training attributes.
            :rtype: numpy.ndarray
        """
        return self._x_train

    def get_y_train(self):
        """ Returns the training labels.
            :return: The training labels.
            :rtype: numpy.ndarray
        """
        return self._y_train

    # Methods
    def distance(self, x1, x2):
        """ Computes the Euclidean distance between two points.
            :param x1: The first point.
            :type x1: numpy.ndarray
            :param x2: The second point.
            :type x2: numpy.ndarray
            :return: The Euclidean distance between the two points.
            :rtype: float
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def get_k_nearest_neighbors(self, x_test, k):
        """ Finds the k nearest neighbors of a test point.
            :param x_test: The test point.
            :type x_test: numpy.ndarray
            :param k: The number of nearest neighbors.
            :type k: int
            :return: list of tuples [(distance, label)]
            :rtype: list of tuples
        """
        neighbors = []

        # Compute the distance from the test point to each training point
        differences = self.get_x_train() - x_test
        distances = np.sqrt(np.sum(differences ** 2, axis=1))

        # Sort the distance from smallest to largest
        sorted_indices = np.argsort(distances, kind='stable')[:k]

        neighbors = []

        # Stores the distance and label of the k nearest neighbors
        for index in sorted_indices:
            class_label = self.get_y_train()[index]
            neighbors.append((distances[index], class_label))

        # Return the first k neighbors
        return neighbors
    def predict(self, x_test, k):
        """ Predicts the class of a test point based on majority vote.
            :param x_test: The test point.
            :type x_test: numpy.ndarray
            :param k: The number of nearest neighbors.
            :type k: int
            :return: The predicted class label.
            :rtype: int
        """
        neighbors = self.get_k_nearest_neighbors(x_test, k)

        class_counts = {}

        # Count the class labels of the k nearest neighbors
        for (_, label) in neighbors:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Find the class label with the maximum count
        predicted_class = max(class_counts, key=class_counts.get)

        return predicted_class

# Wine_KNN_Classification
class Wine_Knn_Classification(Knn_Classification):
    # Constructor
    def __init__(self, filename, train_fraction=0.7, max_k=50, seed=0):
        """ Constructs a Wine Knn classification object.
            :param filename: Path to the White Wine Quality dataset.
            :type filename: str
            :param train_fraction: Fraction of data used for training.
            :type train_fraction: float
            :param max_k: Maximum value of k.
            :type max_k: int
            :param seed: Random seed for reproducibility.
            :type seed: int
            :return: None
        """
        # Data fields
        self._filename = filename
        self._train_fraction = train_fraction
        self._max_k = max_k
        self._seed = seed

        self._attribute_names = None
        self._x_train = None
        self._y_train = None

        self._x_test = None
        self._y_test = None

        self._train_errors = []
        self._test_errors = []

        # Loads dataset
        x_train, y_train = self.prepare_dataset()

        # Initialize the Knn_Classification class
        Knn_Classification.__init__(self, x_train, y_train)
    # Getters
    def get_attribute_names(self):
        """ Returns the attribute names.
            :return: The attribute names.
            :rtype: list of str
        """
        return self._attribute_names
    def get_x_train(self):
        """ Returns the training attributes.
            :return: The training attributes.
            :rtype: numpy.ndarray
        """
        return self._x_train
    def get_y_train(self):
        """ Returns the training labels.
            :return: The training labels.
            :rtype: numpy.ndarray
        """
        return self._y_train
    def get_x_test(self):
        """ Returns the testing attributes.
            :return: The testing attributes.
            :rtype: numpy.ndarray
        """
        return self._x_test
    def get_y_test(self):
        """ Returns the testing labels.
            :return: The testing labels.
            :rtype: numpy.ndarray
        """
        return self._y_test
    # Setters
    def set_attribute_names(self, attribute_names):
        """ Sets the attribute names.
            :param attribute_names: The attribute names.
            :type attribute_names: list of str
            :return: None
        """
        self._attribute_names = attribute_names
    def set_x_train(self, x_train):
        """ Sets the training attributes.
            :param x_train: The training attributes.
            :type x_train: numpy.ndarray
            :return: None
        """
        self._x_train = x_train
    def set_y_train(self, y_train):
        """ Sets the training labels.
            :param y_train: The training labels.
            :type y_train: numpy.ndarray
            :return: None
        """
        self._y_train = y_train
    def set_x_test(self, x_test):
        """ Sets the testing attributes.
            :param x_test: The testing attributes.
            :type x_test: numpy.ndarray
            :return: None
        """
        self._x_test = x_test
    def set_y_test(self, y_test):
        """ Sets the testing labels.
            :param y_test: The testing labels.
            :type y_test: numpy.ndarray
            :return: None
        """
        self._y_test = y_test
    # Methods
    def load_dataset(self):
        """ Loads the White Wine Quality dataset.
            :return: The attributes and class labels.
            :rtype: tuple (numpy.ndarray, numpy.ndarray)
        """
        # Load the dataset
        data = pd.read_csv(self._filename, sep=';')

        # Uses all 11 input attributes
        self.set_attribute_names([
            column for column in data.columns
            if column != 'quality'
        ])
        self.set_x_train(data[self.get_attribute_names()].to_numpy(dtype=float))
        self.set_y_train(data['quality'].to_numpy(dtype=int))

        return self.get_x_train(), self.get_y_train()
    def split_dataset(self, x, y):
        """ Splits the dataset into training and testing sets.
            :param x: All input attributes.
            :type x: numpy.ndarray
            :param y: All class labels.
            :type y: numpy.ndarray
            :return: The training and testing sets.
            :rtype: tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """
        # Set the random seed for reproducibility
        random.seed(self._seed)

        # Shuffle the dataset
        indices = list(range(len(x)))
        random.shuffle(indices)

        # Split the dataset into training and testing sets
        split_index = int(self._train_fraction * len(indices))

        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        x_train = x[train_indices]
        y_train = y[train_indices]

        x_test = x[test_indices]
        y_test = y[test_indices]

        return x_train, y_train, x_test, y_test
    def standardize_dataset(self, x_train, x_test):
        """ Standardizes all input attributes to zero mean and unit variance.
            :param x_train: The training attributes.
            :type x_train: numpy.ndarray
            :param x_test: The testing attributes.
            :type x_test: numpy.ndarray
            :return: The standardized training and testing attributes.
            :rtype: tuple (numpy.ndarray, numpy.ndarray)
        """
        # Compute the mean and standard deviation of the training set
        means = np.mean(x_train, axis=0)
        standard_deviations = np.std(x_train, axis=0)

        # Avoids division by zero
        standard_deviations[standard_deviations == 0] = 1

        # Standardize the training and testing sets
        x_train_standardized = (x_train - means) / standard_deviations
        x_test_standardized = (x_test - means) / standard_deviations

        return x_train_standardized, x_test_standardized
    def prepare_dataset(self):
        """ Prepares the dataset for KNN classification.
            :return: The standardized training attributes and labels.
            :rtype: tuple (numpy.ndarray, numpy.ndarray)
        """
        # Load the dataset
        x, y = self.load_dataset()

        # Split the dataset into training and testing sets
        x_train, y_train, x_test, y_test = self.split_dataset(x, y)

        # Standardize the training and testing sets
        x_train_standardized, x_test_standardized = self.standardize_dataset(x_train, x_test)

        # Store the standardized training and testing sets
        self.set_x_test(x_test_standardized)
        self.set_y_test(y_test)
        self.set_x_train(x_train_standardized)
        self.set_y_train(y_train)

        return self._x_train, self._y_train
    def calculate_errors_for_dataset(self, x_data, y_data):
        """ Calculates the training and testing errors for k = 1 to max_k.
            :param x_data: The input attributes.
            :type x_data: numpy.ndarray
            :param y_data: The class labels.
            :type y_data: numpy.ndarray
            :return: Classification errors for all k values.
            :rtype: tuple (list of float, list of float)
        """
        misclassification_counts = [0] * self._max_k

        for i in range(len(x_data)):
            neighbors = self.get_k_nearest_neighbors(x_data[i], self._max_k)

            class_counts = {}

            for k_index in range(self._max_k):

                label = neighbors[k_index][1]

                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

                # Majority vote to predict the class label
                predicted_class = max(class_counts, key=class_counts.get)

                # Counts the misclassifications
                if predicted_class != y_data[i]:
                    misclassification_counts[k_index] += 1

        # Calculate the classification errors for all k values
        errors = []

        for count in misclassification_counts:
            errors.append(count / len(x_data))

        return errors
    def calculate_errors(self):
        """ Calculates the training and testing errors.
            :return: None
        """
        # Training errors
        # Training set itself is used as the query set
        self._train_errors = self.calculate_errors_for_dataset(self.get_x_train(), self.get_y_train())

        # Testing errors
        self._test_errors = self.calculate_errors_for_dataset(self.get_x_test(), self.get_y_test())
    def print_dataset_info(self):
        """ Prints the dataset information.
            :return: None
        """
        print("White Wine Quality Dataset Information:")
        print(f"Number of records: {len(self.get_x_train()) + len(self.get_x_test())}")
        print(f"Number of attributes: {len(self.get_attribute_names())}")
        print(f"Classes:"
              f" {sorted(np.unique(self.get_y_train()).tolist())}")
        print(f"Training records: "
              f"{len(self.get_x_train())}")
        print(f"Testing records: "
              f"{len(self.get_x_test())}")
    def print_results(self):
        """ Prints the training and testing errors for all k values.
            :return: None
        """
        print("\nKNN Classification Results:")
        print(f"{'k':<5}{'Training Error':<20}{'Testing Error':<20}")
        for k in range(1, self._max_k + 1):
            print(f"{k:<5}{self._train_errors[k - 1]:<20.4f}{self._test_errors[k - 1]:<20.4f}")
    def plot_errors(self):
        """ Plots the training and testing errors for all k values.
            :return: None
        """
        k_values = list(range(1, self._max_k + 1))
        plt.figure(figsize=(9, 6))

        # Training error plot
        plt.plot(k_values, self._train_errors, label='Training Error', marker='o', markersize=3, color='blue')

        # Testing error plot
        plt.plot(k_values, self._test_errors, label='Testing Error', marker='s', markersize=3, color='red')

        plt.title('KNN Classification Errors for White Wine Quality Dataset')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Error rate')
        plt.xticks(range(1, self._max_k + 1, 5))
        plt.ylim(0, 0.55)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig('KNN_Classification_Errors.png',dpi=300, bbox_inches='tight')
        plt.show()
    def run(self):
        """ Runs the KNN classification process.
            :return: None
        """
        # Print dataset information
        self.print_dataset_info()

        # Calculate training and testing errors
        self.calculate_errors()

        # Print results
        self.print_results()

        # Plot errors
        self.plot_errors()
def main():
    # Variable declarations
    filename = "winequality-white.csv"
    train_fraction = 0.7
    max_k = 50
    seed = 0

    # Create a Wine_Knn_Classification object
    wine_knn = Wine_Knn_Classification(filename, train_fraction, max_k, seed)

    # Runs the experiment
    wine_knn.run()

if __name__ == "__main__":
    main()