# Customer Churn Prediction

This project uses a neural network to predict whether a bank customer is likely to leave the bank. The analysis and model training are implemented in the Jupyter notebook [`customer-churn-prediction.ipynb`](./customer-churn-prediction.ipynb), using the dataset [`Churn_Modelling.csv`](./Churn_Modelling.csv).

## Project Overview

The model predicts the `Exited` target:

- `0` — customer stayed
- `1` — customer exited

The dataset contains 10,000 customer records and 14 columns. It includes customer demographics, credit information, account activity, and the churn label.

## Dataset Features

The original dataset includes:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target)

`RowNumber`, `CustomerId`, and `Surname` are removed during preprocessing because they do not provide useful predictive information for this model.

## Workflow

The notebook follows these steps:

1. Load and inspect the dataset.
2. Check for duplicate records and review class distributions.
3. Remove identifier and surname columns.
4. Convert the categorical `Geography` and `Gender` columns using one-hot encoding.
5. Separate the features (`X`) from the target (`y`).
6. Split the data into training and test sets using an 80/20 split.
7. Standardize the numeric input features with `StandardScaler`.
8. Train a TensorFlow/Keras feed-forward neural network:
   - Dense layer with 11 neurons and ReLU activation
   - Dense layer with 11 neurons and ReLU activation
   - Output layer with 1 neuron and sigmoid activation
9. Train for 100 epochs using the Adam optimizer and binary cross-entropy loss.
10. Convert predicted probabilities to class labels using a 0.5 threshold.
11. Evaluate the model with accuracy.

## Result

The notebook records a test accuracy of **86.4%** for the trained model. Results can vary slightly when the notebook is rerun because neural-network training can be affected by initialization and environment settings.

## Requirements

Install the main Python dependencies with:

```bash
pip install numpy pandas scikit-learn tensorflow jupyter
```

Python 3.9 or newer is recommended. TensorFlow availability can vary by operating system and Python version; consult the TensorFlow installation guide if the package does not install in your environment.

## Running the Project

1. Clone or download this project.
2. Install the dependencies.
3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open [`customer-churn-prediction.ipynb`](./customer-churn-prediction.ipynb).
5. Ensure the CSV path in the data-loading cell points to the local [`Churn_Modelling.csv`](./Churn_Modelling.csv) file.
6. Run the notebook from top to bottom.

The notebook currently contains a Kaggle-specific path:

```python
df = pd.read_csv("/kaggle/input/datasets/rjmanoj/credit-card-customer-churn-prediction/Churn_Modelling.csv")
```

When running locally, replace it with a relative path such as:

```python
df = pd.read_csv("Churn_Modelling.csv")
```

## Repository Structure

```text
.
├── Churn_Modelling.csv
├── customer-churn-prediction.ipynb
└── README.md
```

## Notes

This is an educational machine-learning project. The reported accuracy is a single evaluation result and should not be treated as a production performance guarantee. A production system would also require additional validation, model monitoring, threshold selection, and consideration of class imbalance and business costs.
