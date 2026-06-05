# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03 - Silver Layer Creation
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to transform the raw dataset into a clean and standardized Silver Layer.
# MAGIC
# MAGIC The Silver Layer serves as the trusted analytical dataset used for feature engineering and downstream business analytics.

# COMMAND ----------

df = spark.table("default.t_1")

display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Column Standardization
# MAGIC
# MAGIC Raw datasets often contain:
# MAGIC
# MAGIC - Spaces
# MAGIC - Special characters
# MAGIC - Non-standard naming conventions
# MAGIC
# MAGIC To improve maintainability and Spark SQL compatibility, all columns are renamed using snake_case notation.

# COMMAND ----------

df_clean = (
    df
    .withColumnRenamed("Date/Time", "datetime")
    .withColumnRenamed("LV ActivePower (kW)", "active_power_kw")
    .withColumnRenamed("Wind Speed (m/s)", "wind_speed_ms")
    .withColumnRenamed("Theoretical_Power_Curve (KWh)", "theoretical_power_kwh")
    .withColumnRenamed("Wind Direction (°)", "wind_direction_deg")
)

display(df_clean.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Datetime Standardization
# MAGIC
# MAGIC Time information is a critical component of industrial analytics.
# MAGIC
# MAGIC Converting text-based timestamps into proper timestamp data types enables:
# MAGIC
# MAGIC - Time aggregations
# MAGIC - Window functions
# MAGIC - Trend analysis
# MAGIC - Time-series feature engineering

# COMMAND ----------

from pyspark.sql.functions import to_timestamp

df_clean = df_clean.withColumn(
    "datetime",
    to_timestamp("datetime", "dd MM yyyy HH:mm")
)

display(df_clean.limit(5))

# COMMAND ----------

df_clean.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Duplicate Removal
# MAGIC
# MAGIC Duplicate observations may distort analytical results.
# MAGIC
# MAGIC This step removes duplicate records to ensure dataset consistency and reliability.

# COMMAND ----------

before = df_clean.count()

df_clean = df_clean.dropDuplicates()

after = df_clean.count()

print("Rows before:", before)
print("Rows after:", after)
print("Duplicates removed:", before - after)

# COMMAND ----------

# MAGIC %md
# MAGIC # Missing Value Handling
# MAGIC
# MAGIC Incomplete observations may affect:
# MAGIC
# MAGIC - Statistical calculations
# MAGIC - Machine learning performance
# MAGIC - Business insights
# MAGIC
# MAGIC Rows containing missing values are removed to improve overall data quality.

# COMMAND ----------

before = df_clean.count()

df_clean = df_clean.dropna()

after = df_clean.count()

print("Rows before:", before)
print("Rows after:", after)
print("Null rows removed:", before - after)

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver Layer Creation
# MAGIC
# MAGIC After completing all standardization and cleaning steps, the dataset is stored as a reusable Silver Layer.
# MAGIC
# MAGIC This layer will be used as the primary source for feature engineering and analytics development.

# COMMAND ----------

df_clean.write.mode("overwrite").saveAsTable("default.wind_turbine_silver")

print("Silver table created successfully.")

# COMMAND ----------

silver_df = spark.table("default.wind_turbine_silver")

display(silver_df.limit(10))
print("Rows:", silver_df.count())
print("Columns:", len(silver_df.columns))
silver_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver Layer Summary
# MAGIC
# MAGIC Completed Activities:
# MAGIC
# MAGIC - Column standardization
# MAGIC - Datetime conversion
# MAGIC - Duplicate removal
# MAGIC - Missing value handling
# MAGIC
# MAGIC Output Asset:
# MAGIC
# MAGIC default.wind_turbine_silver
# MAGIC
# MAGIC The Silver Layer represents the trusted analytical dataset for the remainder of the project.
# MAGIC
# MAGIC Next Step:
# MAGIC
# MAGIC Feature Engineering and Gold Layer Creation.