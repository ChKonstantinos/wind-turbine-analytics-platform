# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02 - Data Understanding & Quality Assessment
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to understand the dataset structure and assess overall data quality before performing transformations.
# MAGIC
# MAGIC This phase follows the Data Understanding stage of the CRISP-DM methodology.

# COMMAND ----------

df = spark.table("default.t_1")

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Dataset Structure
# MAGIC
# MAGIC The dataset contains SCADA measurements collected from a wind turbine.
# MAGIC
# MAGIC Key variables include:
# MAGIC
# MAGIC - Wind Speed
# MAGIC - Wind Direction
# MAGIC - Active Power Output
# MAGIC - Theoretical Power Output
# MAGIC - Timestamp
# MAGIC
# MAGIC These variables provide the foundation for performance analysis and feature engineering.

# COMMAND ----------

print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

for col_name in df.columns:
    print(col_name)

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC # Descriptive Statistics
# MAGIC
# MAGIC Understanding the distribution of each variable helps identify:
# MAGIC
# MAGIC - Data quality issues
# MAGIC - Extreme values
# MAGIC - Sensor anomalies
# MAGIC - Operational patterns
# MAGIC
# MAGIC The results provide an initial view of turbine behaviour.

# COMMAND ----------

# MAGIC %md
# MAGIC # Missing Values Analysis
# MAGIC
# MAGIC Missing values can negatively impact:
# MAGIC
# MAGIC - Aggregations
# MAGIC - Feature engineering
# MAGIC - Machine learning models
# MAGIC
# MAGIC The objective is to identify whether data completeness issues exist before creating the Silver Layer.

# COMMAND ----------

from pyspark.sql.functions import col, count, when

missing_values = df.select(
    [
        count(
            when(col(c).isNull(), c)
        ).alias(c)
        for c in df.columns
    ]
)

display(missing_values)

# COMMAND ----------

# MAGIC %md
# MAGIC # Duplicate Records Analysis
# MAGIC
# MAGIC Duplicate records can introduce bias into calculations and model training.
# MAGIC
# MAGIC This step verifies whether duplicate observations exist within the dataset.

# COMMAND ----------

total_rows = df.count()

unique_rows = df.dropDuplicates().count()

print("Total Rows:", total_rows)
print("Unique Rows:", unique_rows)
print("Duplicates:", total_rows - unique_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC # Temporal Coverage
# MAGIC
# MAGIC Understanding the available time range is critical for:
# MAGIC
# MAGIC - Time-series analysis
# MAGIC - Trend detection
# MAGIC - Seasonal behaviour identification
# MAGIC - Future forecasting initiatives

# COMMAND ----------

from pyspark.sql.functions import min, max

display(
    df.select(
        min("Date/Time").alias("start_date"),
        max("Date/Time").alias("end_date")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Quality Summary
# MAGIC
# MAGIC Completed Activities:
# MAGIC
# MAGIC - Dataset structure review
# MAGIC - Descriptive statistics analysis
# MAGIC - Missing value assessment
# MAGIC - Duplicate record assessment
# MAGIC - Temporal coverage validation
# MAGIC
# MAGIC Key Outcome:
# MAGIC
# MAGIC The dataset has been assessed and is ready for standardization and cleaning.
# MAGIC
# MAGIC Next Step:
# MAGIC
# MAGIC Silver Layer Creation.