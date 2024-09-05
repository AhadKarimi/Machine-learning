# How to Predict House Prices Using Machine Learning and Python
In this tutorial, we’ll use Python and a machine learning technique called  **Linear Regression**  to predict house prices based on some key features like the number of bedrooms, area, and crime rate in the area. Let’s break this down into easy steps.

## 1. Importing Libraries

First, we need some tools (libraries) to help us:

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
```

**pandas**  helps us manage our data.  
**train_test_split**  divides the data into training and testing sets.  
**LinearRegression**  creates the model that will predict house prices.  
**mean_squared_error**  and  **mean_absolute_error**  check how good our model is.

## 2. Loading the Data

We load our dataset  **[Housing.csv](https://datadp.com/wp-content/uploads/2024/09/Housing.csv)** (right click and 'Save link as...') with information about houses:

```
data = pd.read_csv("Housing.csv")
```
Our data might look like this:

| Price   | Bedroom | Area | Crime |
|---------|---------|------|-------|
| 1000000 | 1       | 100  | 0     |
| 1000000 | 1       | 800  | 0     |
| 500000  | 1       | 100  | 1     |
| 500000  | 1       | 800  | 1     |
| 2000000 | 2       | 200  | 0     |

## 3. Defining Inputs and Output

Next, we pick what features we’ll use to predict the price (called  `X`), and our target (price) is  `y`.

```
X = data[['bedroom', 'area', 'crime']]  # The features we’ll use
y = data['price']  # The price we want to predict
```

## 4. Splitting the Data

We split the data into two parts:

-   **Training set**: Used to train the model (80% of the data).
-   **Test set**: Used to test how well the model performs (20%).

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 5. Building the Model

We create a simple  **Linear Regression**  model:

```
model = LinearRegression()
```

## 6. Training the Model

Now we train the model using the training data:

```
model.fit(X_train, y_train)
```

## 7. Making Predictions

After training, we use the model to predict house prices from our test data:

```
y_pred = model.predict(X_test)
```

## 8. Checking the Model’s Accuracy

We check how good our predictions are using two common methods:

**Mean Squared Error (MSE)**: Squared average of errors (again, lower is better).

**Mean Absolute Error (MAE)**: Average of the errors (lower is better).

```
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE: {mae}")
print(f"MSE: {mse}")
```

## 9. Predicting a New House Price

Finally, we can use the model to predict the price of a new house. Let’s say the house has  **3 bedrooms**,  **700 sqft of area**, and a  **crime rate of 1**:

```
my_predict = model.predict(pd.DataFrame({'bedroom': [3], 'area': [700], 'crime' : [1]}))
print(f"Price: ${my_predict[0]:,.2f}")
```

If everything is set up correctly, it will predict the price based on those values.

## Complete Code:

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("Housing.csv")
#data.head()

X = data[['bedroom', 'area', 'crime']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE: {mae}")
print(f"MSE: {mse}")

my_predict = model.predict(pd.DataFrame({'bedroom': [3], 'area': [300], 'crime' : [1]}))
print(f"Price: ${my_predict[0]:,.2f}")
```
