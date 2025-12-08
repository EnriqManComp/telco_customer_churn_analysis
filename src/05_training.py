import os
import mlflow
import mlflow.spark
import pandas as pd
from pyspark.sql.functions import col
import numpy as np

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import xgboost as xgb
from pyspark.ml.feature import VectorAssembler

training_trigger_dict = dbutils.tasksValues.get(taskKey="Training", value="training_trigger")
training_trigger = training_trigger_dict['training']

if training_trigger == True:
    data = spark.read.table("workspace.telco.ml_training_data")
    # Selecting features
    feature_cols = [c for c in data.columns if c not in ["Churn", "customerID"]]
    # Assembler
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    # Create the assembled DataFrame
    assembled_data = assembler.transform(data).select("customerID", "features", "Churn")
    # Splitting the data
    train, test = assembled_data.randomSplit([0.8, 0.2], seed=42)
    # Convert train to pandas
    train_df = train.select("features", "Churn").toPandas()
    # Create DMatrix 
    dtrain = xgb.DMatrix(train_df["features"].tolist(), label=train_df["Churn"])
    X = np.vstack(train_df["features"].values)
    y = train_df["Churn"].values

    import optuna
    from optuna.integration import XGBoostPruningCallback
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import numpy as np

    # Split training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class imbalance weight
    scale_pos_weight = round(train_df.groupby(["Churn"])['Churn'].count()[0] / train_df.groupby(["Churn"])['Churn'].count()[1],4)

    # Convert to DMatrix for speed
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)


    def objective(trial):

        param = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": trial.suggest_float("eta", 0.1, 0.7),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "scale_pos_weight": scale_pos_weight,
            "verbosity": 0
        }

        pruning_callback = XGBoostPruningCallback(
            trial, "validation-logloss"
        )

        bst = xgb.train(
            param,
            dtrain,
            num_boost_round=1000,
            evals=[(dvalid, "validation")],
            callbacks=[pruning_callback],
            early_stopping_rounds=50
        )

        preds = bst.predict(dvalid)
        return roc_auc_score(y_valid, preds)


    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=10, interval_steps=1
        )
    )

    study.optimize(objective, n_trials=50)

    print("Best trial:", study.best_trial.params)

    scale_pos_weight = round(train_df.groupby(["Churn"])['Churn'].count()[0] / train_df.groupby(["Churn"])['Churn'].count()[1],4)

    params = {
        "objective": "binary:logistic",
        "eta": study.best_trial.params["eta"],
        "max_depth": study.best_trial.params["max_depth"],
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight
    }

    fitted_xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    # Convert to pandas
    test_df = test.toPandas()

    # Convert vectors → arrays for XGBoost
    X_test = test_df["features"].apply(lambda v: v.toArray()).tolist()
    y_test = test_df["Churn"]

    # Create DMatrix
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Performing predictions
    xgb_pred_proba = fitted_xgb_model.predict(dtest)
    threshold = 0.2658
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

    # Compute ROC-AUC
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="probability",
        metricName="areaUnderROC"
    )

    xgb_roc_auc = evaluator.evaluate(xgb_predictions_spark)
    print("XGB-ROC-AUC:", xgb_roc_auc)

    # Save ROC-AUC
    dbutils.fs.put(
        "/dbfs/FileStore/tables/xgb_roc_auc.txt",
        str(xgb_roc_auc),
        overwrite=True
    )

    # Getting training data
    train_data = assembled_data.select("features", "Churn").toPandas()
    # Compute scaled weights
    scale_pos_weight = round(train_data.groupby(["Churn"])["Churn"].count()[0] / train_data.groupby(["Churn"])["Churn"].count()[1],4)
    # Create DMatrix
    dtrain = xgb.DMatrix(train_data["features"].tolist(), label=train_data["Churn"])
    # XGBoost params
    params = {
        "objective": "binary:logistic",
        "eta": study.best_trial.params["eta"],
        "max_depth": study.best_trial.params["max_depth"],    
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight
    }
    # Train XGBoost
    fitted_xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    import pickle
    # Saving model
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(fitted_xgb_model, f)
    













    


  