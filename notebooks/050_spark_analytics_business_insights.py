# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05 - Spark Analytics & Business Insights
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to analyze the Gold Layer and extract business insights related to wind turbine performance.
# MAGIC
# MAGIC This notebook focuses on:
# MAGIC
# MAGIC - Efficiency analysis
# MAGIC - Underperformance detection
# MAGIC - Monthly performance trends
# MAGIC - Wind speed impact
# MAGIC - Operational KPI reporting
# MAGIC
# MAGIC The goal is to translate engineered features into actionable business understanding.

# COMMAND ----------

gold_df = spark.table("default.wind_turbine_gold")

display(gold_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Overall Performance KPIs
# MAGIC
# MAGIC This section calculates high-level KPIs that summarize the turbine's operational performance.
# MAGIC
# MAGIC Key metrics include:
# MAGIC
# MAGIC - Average actual power generation
# MAGIC - Average theoretical power generation
# MAGIC - Average efficiency ratio
# MAGIC - Total performance gap
# MAGIC - Number of underperformance events

# COMMAND ----------

from pyspark.sql.functions import avg, sum, count, col

kpi_df = gold_df.agg(
    avg("active_power_kw").alias("avg_actual_power_kw"),
    avg("theoretical_power_kwh").alias("avg_theoretical_power_kwh"),
    avg("efficiency_ratio").alias("avg_efficiency_ratio"),
    sum("power_gap_kwh").alias("total_power_gap_kwh"),
    sum("underperformance_flag").alias("underperformance_events"),
    count("*").alias("total_records")
)

display(kpi_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Monthly Performance Trends
# MAGIC
# MAGIC Monthly aggregation helps identify whether turbine performance changes over time.
# MAGIC
# MAGIC This can reveal:
# MAGIC
# MAGIC - Seasonal performance patterns
# MAGIC - Operational degradation
# MAGIC - Periods with increased underperformance
# MAGIC - Potential maintenance windows

# COMMAND ----------

monthly_df = (
    gold_df
    .groupBy("year", "month")
    .agg(
        avg("active_power_kw").alias("avg_actual_power_kw"),
        avg("theoretical_power_kwh").alias("avg_theoretical_power_kwh"),
        avg("efficiency_ratio").alias("avg_efficiency_ratio"),
        sum("power_gap_kwh").alias("total_power_gap_kwh"),
        sum("underperformance_flag").alias("underperformance_events"),
        count("*").alias("records")
    )
    .orderBy("year", "month")
)

display(monthly_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Underperformance Analysis
# MAGIC
# MAGIC Underperformance events indicate periods where actual generation falls significantly below expected theoretical output.
# MAGIC
# MAGIC These events may be related to:
# MAGIC
# MAGIC - Mechanical inefficiencies
# MAGIC - Curtailment
# MAGIC - Sensor issues
# MAGIC - Maintenance needs
# MAGIC - Unfavorable operating conditions

# COMMAND ----------

from pyspark.sql.functions import round

underperformance_rate_df = (
    gold_df
    .agg(
        count("*").alias("total_records"),
        sum("underperformance_flag").alias("underperformance_events")
    )
    .withColumn(
        "underperformance_rate",
        round(col("underperformance_events") / col("total_records"), 4)
    )
)

display(underperformance_rate_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Hourly Performance Pattern
# MAGIC
# MAGIC Hourly analysis helps identify whether underperformance is concentrated during specific times of the day.
# MAGIC
# MAGIC This can support operational scheduling, inspection planning, and performance monitoring.

# COMMAND ----------

hourly_df = (
    gold_df
    .groupBy("hour")
    .agg(
        avg("active_power_kw").alias("avg_actual_power_kw"),
        avg("theoretical_power_kwh").alias("avg_theoretical_power_kwh"),
        avg("efficiency_ratio").alias("avg_efficiency_ratio"),
        sum("underperformance_flag").alias("underperformance_events"),
        count("*").alias("records")
    )
    .orderBy("hour")
)

display(hourly_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Wind Speed Performance Bands
# MAGIC
# MAGIC Wind speed is one of the most important drivers of wind turbine power generation.
# MAGIC
# MAGIC By grouping wind speed into bands, we can analyze how efficiently the turbine performs under different wind conditions.

# COMMAND ----------

from pyspark.sql.functions import when

wind_band_df = gold_df.withColumn(
    "wind_speed_band",
    when(col("wind_speed_ms") < 3, "0-3 m/s")
    .when((col("wind_speed_ms") >= 3) & (col("wind_speed_ms") < 6), "3-6 m/s")
    .when((col("wind_speed_ms") >= 6) & (col("wind_speed_ms") < 9), "6-9 m/s")
    .when((col("wind_speed_ms") >= 9) & (col("wind_speed_ms") < 12), "9-12 m/s")
    .otherwise("12+ m/s")
)

wind_band_summary = (
    wind_band_df
    .groupBy("wind_speed_band")
    .agg(
        avg("active_power_kw").alias("avg_actual_power_kw"),
        avg("theoretical_power_kwh").alias("avg_theoretical_power_kwh"),
        avg("efficiency_ratio").alias("avg_efficiency_ratio"),
        sum("power_gap_kwh").alias("total_power_gap_kwh"),
        sum("underperformance_flag").alias("underperformance_events"),
        count("*").alias("records")
    )
    .orderBy("wind_speed_band")
)

display(wind_band_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # Highest Performance Loss Periods
# MAGIC
# MAGIC This section identifies the individual records with the highest gap between theoretical and actual power generation.
# MAGIC
# MAGIC These periods are important because they represent the strongest candidates for operational investigation.

# COMMAND ----------

top_loss_events = (
    gold_df
    .select(
        "datetime",
        "active_power_kw",
        "theoretical_power_kwh",
        "power_gap_kwh",
        "efficiency_ratio",
        "wind_speed_ms",
        "wind_direction_deg"
    )
    .orderBy(col("power_gap_kwh").desc())
)

display(top_loss_events.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC # Business Insights Summary
# MAGIC
# MAGIC Based on the Gold Layer analysis, the project can now support turbine performance monitoring through:
# MAGIC
# MAGIC - Efficiency tracking
# MAGIC - Underperformance detection
# MAGIC - Wind-speed-based performance analysis
# MAGIC - Time-based operational monitoring
# MAGIC - Identification of high-loss periods
# MAGIC
# MAGIC These insights can be used by operations and maintenance teams to prioritize further investigation and reduce energy production losses.