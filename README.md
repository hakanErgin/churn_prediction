# Churn predictions

## Problem

Credit card customers of the bank are leaving the services

[Data set](https://www.kaggle.com/sakshigoyal7/credit-card-customers)

## Solution

- Data exploration and visualization

- Predictions based on this analyzation

## Result

[**Deployed website**](https://infinite-hollows-10453.herokuapp.com/)

---

## Development

Running locally with docker:

```
docker build -t churn-prediction .

docker run -d -v $PWD/:/app -p 8501:8501 churn-prediction
```

## Deployment

`git push heroku <your_branch>:main`

or

`git push heroku main`
