import dagshub
dagshub.init(repo_owner='arafathassan82', repo_name='laptop-price-prediction1', mlflow=True)

import mlflow

mlflow.set_tracking_uri("https://dagshub.com/arafathassan82/laptop-price-prediction1.mlflow/")

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
