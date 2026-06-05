# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04 - Feature Engineering & Gold Layer Creation
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to transform the Silver Layer into a business-ready Gold Layer.
# MAGIC
# MAGIC The Gold Layer contains engineered features and business metrics that support:
# MAGIC
# MAGIC - Performance Monitoring
# MAGIC - Operational Analytics
# MAGIC - Predictive Modeling
# MAGIC - Executive Reporting
# MAGIC
# MAGIC This layer serves as the foundation for advanced analytics and machine learning applications.

# COMMAND ----------

silver_df = spark.table("default.wind_turbine_silver")

display(silver_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Performance Gap Analysis
# MAGIC
# MAGIC A key business objective is to compare actual turbine output against theoretical expectations.
# MAGIC
# MAGIC Performance Gap measures the difference between:
# MAGIC
# MAGIC Actual Power Generation
# MAGIC vs
# MAGIC Expected Power Generation
# MAGIC
# MAGIC This metric helps identify operational inefficiencies and performance losses.

# COMMAND ----------

from pyspark.sql.functions import col

gold_df = silver_df.withColumn(
    "power_gap_kwh",
    col("theoretical_power_kwh") - col("active_power_kw")
)

display(gold_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Efficiency Ratio
# MAGIC
# MAGIC Efficiency Ratio measures how effectively the turbine converts available wind energy into actual power generation.
# MAGIC
# MAGIC Formula:
# MAGIC
# MAGIC Actual Power / Theoretical Power
# MAGIC
# MAGIC Values closer to 1 indicate optimal performance.

# COMMAND ----------

from pyspark.sql.functions import when

gold_df = gold_df.withColumn(
    "efficiency_ratio",
    when(
        col("theoretical_power_kwh") > 0,
        col("active_power_kw") / col("theoretical_power_kwh")
    ).otherwise(None)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Time-Based Features
# MAGIC
# MAGIC Operational behaviour often varies by:
# MAGIC
# MAGIC - Hour of day
# MAGIC - Month
# MAGIC - Season
# MAGIC
# MAGIC Time-based features enable trend analysis and future forecasting initiatives.

# COMMAND ----------

from pyspark.sql.functions import (
    year,
    month,
    dayofmonth,
    hour
)

gold_df = (
    gold_df
    .withColumn("year", year("datetime"))
    .withColumn("month", month("datetime"))
    .withColumn("day", dayofmonth("datetime"))
    .withColumn("hour", hour("datetime"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Underperformance Detection
# MAGIC
# MAGIC An underperformance event occurs when actual power generation falls significantly below theoretical expectations.
# MAGIC
# MAGIC This metric helps prioritize operational investigations and maintenance activities.

# COMMAND ----------

from pyspark.sql.functions import when

gold_df = gold_df.withColumn(
    "underperformance_flag",
    when(
        col("efficiency_ratio") < 0.80,
        1
    ).otherwise(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Gold Layer Creation
# MAGIC
# MAGIC The Gold Layer contains:
# MAGIC
# MAGIC - Cleaned data
# MAGIC - Engineered features
# MAGIC - Business KPIs
# MAGIC - Monitoring indicators
# MAGIC
# MAGIC This dataset is optimized for analytics, dashboards and machine learning workflows.

# COMMAND ----------

gold_df.write.mode("overwrite").saveAsTable(
    "default.wind_turbine_gold"
)

print("Gold Layer created successfully.")

# COMMAND ----------

gold = spark.table("default.wind_turbine_gold")

display(gold.limit(10))

print("Rows:", gold.count())
print("Columns:", len(gold.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC # Gold Layer Summary
# MAGIC
# MAGIC Completed Activities:
# MAGIC
# MAGIC - Performance Gap calculation
# MAGIC - Efficiency Ratio creation
# MAGIC - Time feature extraction
# MAGIC - Underperformance detection
# MAGIC - Gold Layer creation
# MAGIC
# MAGIC Output Asset:
# MAGIC
# MAGIC default.wind_turbine_gold
# MAGIC
# MAGIC The Gold Layer is now ready for:
# MAGIC
# MAGIC - Operational Analytics
# MAGIC - Executive Reporting
# MAGIC - Predictive Modeling
# MAGIC - Recommendation Engine Development