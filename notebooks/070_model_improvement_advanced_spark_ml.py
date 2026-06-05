# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 07 - Model Improvement & Advanced Spark ML
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to improve the baseline model by comparing multiple Spark ML regression algorithms.
# MAGIC
# MAGIC Models evaluated:
# MAGIC
# MAGIC - Linear Regression
# MAGIC - Random Forest Regressor
# MAGIC - Gradient Boosted Trees Regressor
# MAGIC
# MAGIC The objective is to identify the model that best predicts wind turbine power output based on environmental and temporal features.

# COMMAND ----------

# gold_df = spark.table("default.wind_turbine_gold")
gold_df = spark.table("default.wind_turbine_gold_advanced")

display(gold_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Selection
# MAGIC
# MAGIC The same feature set is used across all models to ensure a fair comparison.
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

from pyspark.ml.feature import VectorAssembler

# model_df = gold_df.select(
#     "wind_speed_ms",
#     "wind_direction_deg",
#     "hour",
#     "month",
#     "active_power_kw"
# ).dropna()

model_df = gold_df.select(
    "wind_speed_ms",
    "wind_direction_deg",
    "hour",
    "month",
    "prev_wind_speed",
    "prev_active_power",
    "avg_wind_speed_1h",
    "avg_power_1h",
    "wind_category",
    "season",
    "active_power_kw"
).dropna()

# assembler = VectorAssembler(
#     inputCols=[
#         "wind_speed_ms",
#         "wind_direction_deg",
#         "hour",
#         "month"
#     ],
#     outputCol="features"
# )

# assembler = VectorAssembler(
#     inputCols=[
#         "wind_speed_ms",
#         "wind_direction_deg",
#         "hour",
#         "month",
#         "prev_wind_speed",
#         "prev_active_power",
#         "avg_wind_speed_1h",
#         "avg_power_1h"
#     ],
#     outputCol="features"
# )

# model_df = assembler.transform(model_df).select(
#     "features",
#     "active_power_kw"
# )

# display(model_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Categorical Feature Engineering
# MAGIC
# MAGIC Some business features are categorical and cannot be directly used by Spark ML algorithms.
# MAGIC
# MAGIC The following transformations are applied:
# MAGIC
# MAGIC - StringIndexer
# MAGIC - OneHotEncoder
# MAGIC
# MAGIC Features:
# MAGIC
# MAGIC - wind_category
# MAGIC - season
# MAGIC
# MAGIC This process converts categorical variables into machine-learning-compatible numerical representations.

# COMMAND ----------

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder
)

# COMMAND ----------

wind_indexer = StringIndexer(
    inputCol="wind_category",
    outputCol="wind_category_idx"
)

season_indexer = StringIndexer(
    inputCol="season",
    outputCol="season_idx"
)

# COMMAND ----------

gold_df = wind_indexer.fit(gold_df).transform(gold_df)

gold_df = season_indexer.fit(gold_df).transform(gold_df)

# COMMAND ----------

encoder = OneHotEncoder(
    inputCols=[
        "wind_category_idx",
        "season_idx"
    ],
    outputCols=[
        "wind_category_ohe",
        "season_ohe"
    ]
)

# COMMAND ----------

gold_df = encoder.fit(gold_df).transform(gold_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Vector Creation
# MAGIC
# MAGIC All numerical and encoded categorical features are combined into a single feature vector using Spark's VectorAssembler.

# COMMAND ----------

model_df = gold_df.select(
    "wind_speed_ms",
    "wind_direction_deg",
    "hour",
    "month",
    "prev_wind_speed",
    "prev_active_power",
    "avg_wind_speed_1h",
    "avg_power_1h",
    "wind_category_ohe",
    "season_ohe",
    "active_power_kw"
).dropna()

assembler = VectorAssembler(
    inputCols=[
        "wind_speed_ms",
        "wind_direction_deg",
        "hour",
        "month",
        "prev_wind_speed",
        "prev_active_power",
        "avg_wind_speed_1h",
        "avg_power_1h",
        "wind_category_ohe",
        "season_ohe"
    ],
    outputCol="features"
)

model_df = assembler.transform(model_df).select(
    "features",
    "active_power_kw"
)

display(model_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Train-Test Split
# MAGIC
# MAGIC The data is split into training and testing sets using a fixed random seed for reproducibility.

# COMMAND ----------

train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training
# MAGIC
# MAGIC Three regression models are trained:
# MAGIC
# MAGIC 1. Linear Regression as the baseline
# MAGIC 2. Random Forest Regressor for non-linear relationships
# MAGIC 3. Gradient Boosted Trees Regressor for sequential error correction

# COMMAND ----------

from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor

models = {
    "Linear Regression": LinearRegression(
        featuresCol="features",
        labelCol="active_power_kw"
    ),
    "Random Forest": RandomForestRegressor(
        featuresCol="features",
        labelCol="active_power_kw",
        numTrees=100,
        maxDepth=8,
        seed=42
    ),
    "Gradient Boosted Trees": GBTRegressor(
        featuresCol="features",
        labelCol="active_power_kw",
        maxIter=100,
        maxDepth=6,
        seed=42
    )
}

trained_models = {}

for name, model in models.items():
    trained_models[name] = model.fit(train_df)
    print(f"{name} trained successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation
# MAGIC
# MAGIC Models are evaluated using:
# MAGIC
# MAGIC - RMSE
# MAGIC - MAE
# MAGIC - R²
# MAGIC
# MAGIC The best model should have the lowest RMSE and MAE and the highest R².

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluators = {
    "RMSE": RegressionEvaluator(
        labelCol="active_power_kw",
        predictionCol="prediction",
        metricName="rmse"
    ),
    "MAE": RegressionEvaluator(
        labelCol="active_power_kw",
        predictionCol="prediction",
        metricName="mae"
    ),
    "R2": RegressionEvaluator(
        labelCol="active_power_kw",
        predictionCol="prediction",
        metricName="r2"
    )
}

results = []

for name, model in trained_models.items():
    predictions = model.transform(test_df)

    rmse = evaluators["RMSE"].evaluate(predictions)
    mae = evaluators["MAE"].evaluate(predictions)
    r2 = evaluators["R2"].evaluate(predictions)

    results.append((name, rmse, mae, r2))

results_df = spark.createDataFrame(
    results,
    ["model", "rmse", "mae", "r2"]
)

display(results_df.orderBy("rmse"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Best Model Selection
# MAGIC
# MAGIC The model with the lowest RMSE is selected as the best-performing model.
# MAGIC
# MAGIC This model can be used as the predictive engine for estimating expected turbine power output.

# COMMAND ----------

best_model_row = results_df.orderBy("rmse").first()

best_model_name = best_model_row["model"]

print("Best Model:", best_model_name)
print("RMSE:", best_model_row["rmse"])
print("MAE:", best_model_row["mae"])
print("R2:", best_model_row["r2"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction Output
# MAGIC
# MAGIC The best model is applied to the test dataset to generate predictions.
# MAGIC
# MAGIC These predictions can later be compared against actual production values to identify performance deviations.

# COMMAND ----------

best_model = trained_models[best_model_name]

best_predictions = best_model.transform(test_df)

display(
    best_predictions.select(
        "active_power_kw",
        "prediction"
    ).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Improvement Summary
# MAGIC
# MAGIC Completed Activities:
# MAGIC
# MAGIC - Trained baseline Linear Regression model
# MAGIC - Trained Random Forest Regressor
# MAGIC - Trained Gradient Boosted Trees Regressor
# MAGIC - Compared models using RMSE, MAE and R²
# MAGIC - Selected the best-performing model
# MAGIC
# MAGIC Business Interpretation:
# MAGIC
# MAGIC A stronger predictive model allows the business to estimate expected turbine output under given operating conditions.
# MAGIC
# MAGIC This expected output can later be compared with actual output to detect underperformance and possible operational inefficiencies.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Impact
# MAGIC
# MAGIC Advanced temporal features generated using Spark Window Functions significantly improved predictive performance across all evaluated models.
# MAGIC
# MAGIC The largest improvements were observed in RMSE and MAE, demonstrating that feature engineering contributed more to model quality than algorithm selection alone.
# MAGIC
# MAGIC This finding highlights the importance of domain-aware feature design in industrial analytics applications.

# COMMAND ----------

# MAGIC %md
# MAGIC