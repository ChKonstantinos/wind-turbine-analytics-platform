# Databricks notebook source
# MAGIC %md
# MAGIC # Wind Turbine Analytics Platform (Spark Edition)
# MAGIC
# MAGIC ## Notebook 01 - Spark Setup & Data Ingestion
# MAGIC
# MAGIC ### Project Overview
# MAGIC
# MAGIC This project presents an end-to-end analytics solution for wind turbine performance monitoring using Apache Spark and Databricks.
# MAGIC
# MAGIC The objective is to analyze SCADA sensor data, identify performance losses, engineer business features, and build analytical assets that support operational decision-making.
# MAGIC
# MAGIC ### Technology Stack
# MAGIC
# MAGIC - Apache Spark
# MAGIC - Databricks
# MAGIC - PySpark
# MAGIC - SQL
# MAGIC - Machine Learning
# MAGIC - Data Engineering Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC # Business Problem
# MAGIC
# MAGIC Wind turbines operate continuously under changing environmental conditions.
# MAGIC
# MAGIC Underperformance can lead to:
# MAGIC
# MAGIC - Reduced energy production
# MAGIC - Revenue loss
# MAGIC - Maintenance inefficiencies
# MAGIC - Increased operational costs
# MAGIC
# MAGIC The goal of this project is to identify operational inefficiencies by comparing actual power generation against expected theoretical output.

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Ingestion
# MAGIC
# MAGIC The first step of the analytics pipeline is loading the raw SCADA dataset into Spark.
# MAGIC
# MAGIC Spark allows distributed processing and scalable transformations, making it suitable for industrial-scale energy analytics applications.

# COMMAND ----------

spark

# COMMAND ----------

df = spark.table("workspace.default.t_1")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Validation
# MAGIC
# MAGIC Before building any analytics pipeline, it is important to verify:
# MAGIC
# MAGIC - Dataset dimensions
# MAGIC - Column names
# MAGIC - Data types
# MAGIC - Successful ingestion
# MAGIC
# MAGIC This step ensures data consistency before moving into downstream processing.

# COMMAND ----------

print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.describe())

# COMMAND ----------

spark.sql("SHOW TABLES").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Summary
# MAGIC
# MAGIC Completed Activities:
# MAGIC
# MAGIC - Spark environment initialized
# MAGIC - Dataset loaded into Databricks
# MAGIC - Table created
# MAGIC - Data successfully validated
# MAGIC
# MAGIC Output Asset:
# MAGIC
# MAGIC default.t_1
# MAGIC
# MAGIC Next Step:
# MAGIC
# MAGIC Data Understanding and Quality Assessment.