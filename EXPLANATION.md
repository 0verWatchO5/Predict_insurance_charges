# Linear Regression on Insurance Dataset

## Objective
Predict insurance charges based on a person's age using Simple Linear Regression.

---

## Dataset Overview
- **Source:** insurance.csv
- **Total Records:** 1338
- **Key Columns Used:**
  - `age` — Age of the person (independent variable / X)
  - `charges` — Medical insurance cost (dependent variable / Y)

---

## Steps Performed

### 1. Data Loading & Exploration
- Loaded the dataset using `pandas`
- Checked for null values using `df.isnull().sum()` — no missing values found
- Explored the data using `df.head()`, `df.tail()`, `df.describe()`, and `df.info()`

### 2. Data Visualization
- Plotted a scatter plot of **Age vs Insurance Charges**
- Observation: As age increases, insurance charges tend to increase, showing a positive relationship

### 3. Model Building
- Split the data into **70% training** and **30% testing** sets using `train_test_split`
- Applied **Simple Linear Regression** using `sklearn.linear_model.LinearRegression`
- Fitted the model on the training data

### 4. Model Parameters
- **Intercept (b0):** The base insurance charge when age is 0
- **Slope (b1):** For every 1 year increase in age, charges increase by this amount

### 5. Prediction & Evaluation
- Predicted insurance charges for the test set
- Compared **Actual vs Predicted** charges in a table
- Plotted the **regression line** over the test data scatter plot

### 6. New Prediction
- Used the trained model to predict insurance charges for a given age input

---

## Regression Equation
```
Charges = b0 + b1 × Age
```

---

## Conclusion
The linear regression model shows a **positive correlation** between age and insurance charges. As age increases, the predicted insurance charges also increase. This aligns with real-world expectations, as older individuals tend to have higher medical costs.