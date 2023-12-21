**Report on Titanic Dataset Analysis and Model Building**

## Introduction:
As a machine learning intern, the task was to work with the Titanic dataset, which contains passenger information from the Titanic ship. The goal was to build a predictive model that classifies whether a passenger survived or not based on available features. This is a binary classification problem.

## Approach:
### Data Exploration:

1. **Loading Data:**
   - The dataset was loaded using pandas.

2. **Data Exploration:**
   - A quick overview of the data using `train_data.head()` and `train_data.info()` was performed.
   - Descriptive statistics were obtained using `train_data.describe()`.
   - The presence of missing values was checked using `train_data.isnull().sum()`.

### Data Processing:
1. **Handling Missing Values:**
   - The "Cabin" column, with a significant number of missing values, was dropped.
   - Invalid Age values were removed.
   - Missing values in the "Age" column were imputed with the median.
   - Null values in the "Embarked" column were replaced with the most common category.

2. **Feature Engineering:**
   - Categorical variables "Sex" and "Embarked" were encoded using one-hot encoding.
   - Age and Ticket columns were converted to numeric types.

### Data Visualization:
1. **Correlation Analysis:**
   - Investigated the correlation of features with the target variable "Survived."

2. **Data Visualization:**
   - Histograms were plotted for each feature, comparing the distribution for survivors and non-survivors.

### Model Building:

1. **Feature Selection:**
   - Selected features for the model: ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"].

2. **Model Training:**
   - Split the data into training and testing sets (80% for training, 20% for testing).
   - Scaled the features using Z-score normalization.
   - Chose RandomForestClassifier for model training.

3. **Model Evaluation:**
   - Evaluated the model on the test set, obtaining an accuracy of approximately 79.6%.

### Hyperparameter Tuning:
1. **Grid Search:**
   - Used GridSearchCV to find the best hyperparameters for the RandomForestClassifier.
   - Explored different values for "n_estimators," "max_depth," "min_samples_split," and "min_samples_leaf."

2. **Best Model:**
   - The best hyperparameters were found to be {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 50}.
   - The model with these hyperparameters achieved a test accuracy of approximately 80.99%.

## Conclusion:
The analysis and model building on the Titanic dataset involved data exploration, preprocessing, visualization, and model training. The RandomForestClassifier, after hyperparameter tuning, showed improved accuracy. 
