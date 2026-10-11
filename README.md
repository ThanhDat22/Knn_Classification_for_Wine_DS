# K - Nearest Neighbor Classification for White Wine Quality

## I. Overview

This project was completed for a Data Mining course and demonstrates the implementation of KNN classification using Python. The goal is to predict white wine quality classes based on physicochemical attributes and examine how different values of `k` affect the training and testing classification errors.

## II. Dataset Preparation

### 2.1 White Wine Quality Dataset

The program uses the **White Wine Quality** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).

- Input data file: `winequality-white.csv`
- Input attributes: all 11 physicochemical attributes
- Class label: `quality`
- All quality classes present in the dataset are retained.

### 2.2 Training and Testing Sets

The dataset is randomly shuffled and divided into:

- Training set: `70%` of the records
- Testing set: `30%` of the records
- Random seed: `0` for reproducibility

No separate validation set or cross-validation is used.

### 2.3 Data Standardization

All 11 input attributes are standardized using the mean and standard deviation of the **training set**. The same training-set statistics are used to standardize the testing set. This puts attributes with different numerical scales on a comparable scale before calculating distances.

## III. KNN Classification Method

### 3.1 Euclidean Distance

The program measures the distance between a query point and each training point using Euclidean distance. It sorts these distances and selects the `k` nearest training records.

### 3.2 Majority Vote

Each of the `k` nearest neighbors contributes one vote for its quality class. The class receiving the most votes is used as the predicted class. When classes have equal vote counts, the implementation selects the class that first appears among the nearest neighbors.

### 3.3 Values of k

The program evaluates every integer value from `k = 1` to `k = 50` to study how the number of neighbors affects classification errors.

## IV. Classification Error Evaluation

### 4.1 Training Error

Each training record is classified using the training set itself as the set of possible neighbors. Therefore, a training record can be its own nearest neighbor at distance zero.

### 4.2 Testing Error

Each testing record is classified using neighbors from the training set. The predicted quality class is compared with its actual class label.

### 4.3 Error Calculation and Plot

For both datasets, classification error is calculated as:

`Error = Number of misclassified records / Total number of records`

The program prints the training and testing error rates for every `k` from `1` to `50`. It also plots both errors against `k` on the same figure and saves the figure as `KNN_Classification_Errors.png`.

## V. Observations

- At `k = 1`, the training error is zero because each training point is included as its own nearest neighbor.
- Smaller values of `k` make predictions depend on fewer nearby training records.
- Larger values of `k` allow more training records to influence the majority vote, creating smoother class predictions.
- Comparing training and testing errors across `k` helps illustrate the relationship between model complexity and classification performance.

The exact error trends and the best-performing value of `k` should be determined from the program's output and plot rather than assumed in advance.

## VI. Running the Program

Run the program:

```bash
python Knn_Classification_for_Wine.py
```

The program displays dataset information, prints the training and testing errors for `k = 1` through `50`, and generates `KNN_Classification_Errors.png`.

This project demonstrates how KNN classification can predict wine quality classes and how changing the number of nearest neighbors affects training and testing errors.
