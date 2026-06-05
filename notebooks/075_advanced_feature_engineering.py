# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 07.5 - Advanced Feature Engineering
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to create advanced analytical features using Spark Window Functions.
# MAGIC
# MAGIC These features capture historical operating behaviour and provide additional context for machine learning models.
# MAGIC
# MAGIC Feature engineering is often one of the most important factors influencing predictive model performance.

# COMMAND ----------

gold_df = spark.table("default.wind_turbine_gold")

display(gold_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Window Functions
# MAGIC
# MAGIC Wind turbine performance depends not only on current conditions but also on recent operating history.
# MAGIC
# MAGIC Window functions allow us to create lagged and rolling features that capture temporal behaviour.

# COMMAND ----------

from pyspark.sql.window import Window

from pyspark.sql.functions import (
    lag,
    avg,
    col
)

# COMMAND ----------

window_spec = Window.orderBy("datetime")

# COMMAND ----------

# MAGIC %md
# MAGIC # Lag Features
# MAGIC
# MAGIC Lag features provide information about previous observations.
# MAGIC
# MAGIC Examples:
# MAGIC
# MAGIC - Previous wind speed
# MAGIC - Previous power generation
# MAGIC
# MAGIC These features help models identify short-term trends and momentum.

# COMMAND ----------

gold_df = gold_df.withColumn(
    "prev_wind_speed",
    lag("wind_speed_ms", 1).over(window_spec)
)

# COMMAND ----------

gold_df = gold_df.withColumn(
    "prev_active_power",
    lag("active_power_kw", 1).over(window_spec)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Rolling Average Features
# MAGIC
# MAGIC Rolling averages smooth short-term fluctuations and provide a more stable representation of turbine behaviour.

# COMMAND ----------

rolling_window = (
    Window
    .orderBy("datetime")
    .rowsBetween(-6, 0)
)

# COMMAND ----------

gold_df = gold_df.withColumn(
    "avg_wind_speed_1h",
    avg("wind_speed_ms").over(rolling_window)
)

# COMMAND ----------

gold_df = gold_df.withColumn(
    "avg_power_1h",
    avg("active_power_kw").over(rolling_window)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Wind Speed Categories
# MAGIC
# MAGIC Wind speed categories provide business-friendly operational segments that can support reporting and dashboarding.

# COMMAND ----------

from pyspark.sql.functions import when

gold_df = gold_df.withColumn(
    "wind_category",
    when(col("wind_speed_ms") < 3, "Low")
    .when(col("wind_speed_ms") < 8, "Medium")
    .otherwise("High")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Seasonal Features
# MAGIC
# MAGIC Seasonal variables often capture environmental patterns that influence turbine performance.

# COMMAND ----------

gold_df = gold_df.withColumn(
    "season",
    when(col("month").isin([12,1,2]), "Winter")
    .when(col("month").isin([3,4,5]), "Spring")
    .when(col("month").isin([6,7,8]), "Summer")
    .otherwise("Autumn")
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Advanced Gold Layer
# MAGIC
# MAGIC The enhanced Gold Layer now contains:
# MAGIC
# MAGIC - Historical features
# MAGIC - Rolling statistics
# MAGIC - Operational categories
# MAGIC - Seasonal indicators
# MAGIC
# MAGIC These features improve model explainability and predictive capability.

# COMMAND ----------

gold_df.write.mode("overwrite").saveAsTable(
    "default.wind_turbine_gold_advanced"
)

print("Advanced Gold Layer created successfully.")

# COMMAND ----------

advanced_gold = spark.table(
    "default.wind_turbine_gold_advanced"
)

display(
    advanced_gold.limit(10)
)