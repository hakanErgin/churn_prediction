# churn_prediction

https://www.kaggle.com/sakshigoyal7/credit-card-customers

running locally
docker
docker build -t churn-prediction .
docker run -d -v $PWD/:/app -p 8501:8501 churn-prediction
