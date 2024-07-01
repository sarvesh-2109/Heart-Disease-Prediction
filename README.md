# Heart Disease Prediction

![Heart Disease](https://t3.ftcdn.net/jpg/06/06/29/34/360_F_606293479_9iTncv5OBYwY2RBMsFa6yTmIedXjR1VZ.jpg)

Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve the chances of effective treatment and management. This project aims to develop a machine learning model to predict the likelihood of heart disease based on various health metrics.

## Project Overview

This project involves:

- Data loading and preprocessing
- Feature scaling and one-hot encoding
- Dimensionality reduction using PCA
- Model training using Gradient Boosting Classifier
- Model evaluation
- Saving and loading the model for future predictions
- Simulating user input for real-time prediction

## Dataset

The dataset used in this project is the Heart Disease dataset, which is available in CSV format. It contains various health metrics that can be used to predict the presence of heart disease.

## Libraries Used

- `pandas` for data manipulation and analysis
- `scikit-learn` for machine learning and model evaluation
- `joblib` for model persistence
- `matplotlib` for data visualization

## Installation

To run this project, you need to have Python and the following libraries installed:

```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```

## Data Preprocessing

1. **Loading the dataset:**

    ```python
    data = pd.read_csv('/content/heart.csv')
    ```

2. **Checking for null values:**

    ```python
    data.isnull().sum()
    ```

3. **One-hot encoding:**

    ```python
    data_encoded = pd.get_dummies(data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    ```

4. **Feature scaling:**

    ```python
    scaler = StandardScaler()
    features = data_encoded.drop('target', axis=1)
    scaled_features = scaler.fit_transform(features)
    ```

5. **Combining scaled features with the target:**

    ```python
    processed_data = pd.concat([scaled_features_df, data_encoded['target']], axis=1)
    ```

## Model Training

1. **Splitting the dataset:**

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2. **Applying PCA:**

    ```python
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    ```

3. **Training the Gradient Boosting model:**

    ```python
    gradient_boosting = GradientBoostingClassifier(random_state=42)
    gradient_boosting.fit(X_train_pca, y_train)
    ```

## Model Evaluation

Evaluating the model using various metrics:

```python
y_pred_gradient_boosting = gradient_boosting.predict(X_test_pca)
gradient_boosting_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_gradient_boosting),
    'precision': precision_score(y_test, y_pred_gradient_boosting),
    'recall': recall_score(y_test, y_pred_gradient_boosting),
    'f1_score': f1_score(y_test, y_pred_gradient_boosting),
    'roc_auc': roc_auc_score(y_test, y_pred_gradient_boosting)
}

print("Gradient Boosting Metrics:", gradient_boosting_metrics)
```

## Saving the Model

The model can be saved to disk using `joblib`:

```python
model_file_path = 'gradient_boosting_model.joblib'
dump(gradient_boosting, model_file_path)
```

## Predicting Using User's Input

The model can predict the likelihood of heart disease based on user input:

1. **Loading the saved model:**

    ```python
    gradient_boosting = load(model_file_path)
    ```

2. **Simulating user input and making prediction:**

    ```python
    user_input = {
        'age': int(input("Enter your age: ")),
        'sex': int(input("Enter your sex (1: male, 0: female): ")),
        'cp': int(input("Enter chest pain type (0: typical angina, 1: atypical anginaValue, 2: non-anginal pain, 3: asymptomatic): ")),
        'trestbps': int(input("Enter resting blood pressure (in mm Hg): ")),
        'chol': int(input("Enter serum cholesterol in mg/dl: ")),
        'fbs': int(input("Enter fasting blood sugar > 120 mg/dl (1: true, 0: false): ")),
        'restecg': int(input("Enter resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy): ")),
        'thalach': int(input("Enter maximum heart rate achieved: ")),
        'exang': int(input("Enter exercise induced angina (1: yes, 0: no): ")),
        'oldpeak': float(input("Enter ST depression induced by exercise relative to rest: ")),
        'slope': int(input("Enter the slope of the peak exercise ST segment (0: upsloping, 1: flat 2: downsloping): ")),
        'ca': int(input("Enter number of major vessels colored by fluoroscopy (0-3): ")),
        'thal': int(input("Enter thalassemia (1: normal, 2: fixed defect, 3: reversable defect): "))
    }
    ```

3. **One-hot encoding and scaling the user input:**

    ```python
    input_df = pd.DataFrame(user_input, index=[0])
    input_df_encoded = pd.get_dummies(input_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    for column in features.columns:
        if column not in input_df_encoded.columns:
            input_df_encoded[column] = 0
    input_df_encoded = input_df_encoded[features.columns]
    scaled_input = scaler.transform(input_df_encoded)
    pca_input = pca.transform(scaled_input)
    ```

4. **Making prediction:**

    ```python
    user_prediction = gradient_boosting.predict(pca_input)
    user_prediction_proba = gradient_boosting.predict_proba(pca_input)
    prediction_percentage = user_prediction_proba[0][1] * 100

    if user_prediction[0] == 1:
        print(f"The model predicts that the user is at risk of heart disease with a probability of {prediction_percentage:.2f}%.")
    else:
        print(f"The model predicts that the user is not at risk of heart disease with a probability of {100 - prediction_percentage:.2f}%.")
    ```

## Conclusion

This project demonstrates how to build a machine learning model to predict the likelihood of heart disease. It covers data preprocessing, model training, evaluation, and user input prediction. The model can significantly aid in early detection and management of heart disease.

## Repository

The complete code and documentation for this project can be found on GitHub: [Heart Disease Prediction](https://github.com/sarvesh-2109/Heart-Disease-Prediction)

Your feedback and suggestions are welcome!

# MIT License

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
