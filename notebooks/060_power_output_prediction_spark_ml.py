# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 06 - Power Output Prediction with Spark ML
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to develop a machine learning model capable of predicting wind turbine power generation.
# MAGIC
# MAGIC The model will learn the relationship between:
# MAGIC
# MAGIC - Wind Speed
# MAGIC - Wind Direction
# MAGIC - Hour
# MAGIC - Month
# MAGIC
# MAGIC and the target variable:
# MAGIC
# MAGIC - Active Power Output
# MAGIC
# MAGIC The resulting model can be used for operational forecasting and performance benchmarking.

# COMMAND ----------

gold_df = spark.table("default.wind_turbine_gold")

display(gold_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Selection
# MAGIC
# MAGIC The selected features represent the most important environmental and temporal drivers of turbine performance.
# MAGIC
# MAGIC Features:
# MAGIC
# MAGIC - wind_speed_ms
# MAGIC - wind_direction_deg
# MAGIC - hour
# MAGIC - month
# MAGIC
# MAGIC Target:
# MAGIC
# MAGIC - active_power_kw

# COMMAND ----------

model_df = gold_df.select(
    "wind_speed_ms",
    "wind_direction_deg",
    "hour",
    "month",
    "active_power_kw"
)

display(model_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Vector Creation
# MAGIC
# MAGIC Spark ML models require all features to be combined into a single feature vector.
# MAGIC
# MAGIC VectorAssembler is used to transform multiple columns into a machine-learning-ready format.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[
        "wind_speed_ms",
        "wind_direction_deg",
        "hour",
        "month"
    ],
    outputCol="features"
)

model_df = assembler.transform(model_df)

display(
    model_df.select(
        "features",
        "active_power_kw"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Train-Test Split
# MAGIC
# MAGIC The dataset is divided into:
# MAGIC
# MAGIC - 80% Training Data
# MAGIC - 20% Test Data
# MAGIC
# MAGIC This allows objective evaluation of model performance.

# COMMAND ----------

train_df, test_df = model_df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Train Rows:", train_df.count())
print("Test Rows:", test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Baseline Model
# MAGIC
# MAGIC Linear Regression is used as the initial baseline model.
# MAGIC
# MAGIC The objective is to establish a simple benchmark before testing more advanced algorithms.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol="features",
    labelCol="active_power_kw"
)

lr_model = lr.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Predictions
# MAGIC
# MAGIC The trained model is applied to the test dataset to generate power output predictions.

# COMMAND ----------

predictions = lr_model.transform(test_df)

display(
    predictions.select(
        "active_power_kw",
        "prediction"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation
# MAGIC
# MAGIC Regression models are evaluated using:
# MAGIC
# MAGIC - RMSE (Root Mean Squared Error)
# MAGIC - MAE (Mean Absolute Error)
# MAGIC - R² (Coefficient of Determination)
# MAGIC
# MAGIC Lower error values and higher R² indicate better predictive performance.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

rmse_evaluator = RegressionEvaluator(
    labelCol="active_power_kw",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = rmse_evaluator.evaluate(predictions)

print("RMSE:", rmse)

# COMMAND ----------

mae_evaluator = RegressionEvaluator(
    labelCol="active_power_kw",
    predictionCol="prediction",
    metricName="mae"
)

mae = mae_evaluator.evaluate(predictions)

print("MAE:", mae)

# COMMAND ----------

r2_evaluator = RegressionEvaluator(
    labelCol="active_power_kw",
    predictionCol="prediction",
    metricName="r2"
)

r2 = r2_evaluator.evaluate(predictions)

print("R²:", r2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Performance Summary
# MAGIC
# MAGIC The baseline Linear Regression model establishes an initial benchmark for wind turbine power prediction.
# MAGIC
# MAGIC Future improvements may include:
# MAGIC
# MAGIC - Random Forest Regressor
# MAGIC - Gradient Boosted Trees
# MAGIC - XGBoost
# MAGIC - Feature enrichment
# MAGIC - Hyperparameter tuning
# MAGIC
# MAGIC The model demonstrates how Spark ML can be integrated into an end-to-end industrial analytics workflow.

# COMMAND ----------

# MAGIC %md
# MAGIC