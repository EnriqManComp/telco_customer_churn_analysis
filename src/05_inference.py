from pyspark.ml.feature import VectorAssembler
import pickle
from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import xgboost as xgb
import pandas as pd

data = spark.read.table("workspace.telco.ml_silver_data")
# Saving model
with open('xgb_model.pkl', 'rb') as f:
    fitted_xgb_model = pickle.load(f)

feature_cols = [c for c in data.columns if c not in ["Churn", "customerID"]]
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

final_test_data = data.filter(col("Churn") == "0")
final_test_data = assembler.transform(final_test_data).select("customerID", "features", "Churn")
test_data = final_test_data.select("features", "Churn").toPandas()

data.write.mode("append").saveAsTable("workspace.telco.train_data")

X_test = test_data["features"].tolist()
y_test = test_data["Churn"]

# Create DMatrix
dtest = xgb.DMatrix(X_test)

# Performing predictions
xgb_pred_proba = fitted_xgb_model.predict(dtest)
# "Yes" class portion
threshold = 0.2658
# Predictions
xgb_pred_label = (xgb_pred_proba >= threshold).astype(int)

# Build Spark rows
rows = []
for prob, label, pred in zip(xgb_pred_proba, y_test, xgb_pred_label):
    rows.append(Row(
        label=float(label),
        prediction=float(pred),
        probability=Vectors.dense([1 - prob, prob])
    ))

# Create Spark DataFrame
xgb_predictions_spark = spark.createDataFrame(rows)

# Adding customerID column
xgb_predictions_w_clients = pd.concat([final_test_data.select("customerID").toPandas(), pd.DataFrame(xgb_pred_label)], axis=1)
# Naming columns
xgb_predictions_w_clients.columns = ["customerID", "prediction"]

spark.createDataFrame(xgb_predictions_w_clients).write.mode("overwrite").saveAsTable("workspace.telco.xgb_predictions")



