# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 09 - Executive Dashboard Layer
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to create business-ready dashboard datasets.
# MAGIC
# MAGIC These datasets are optimized for reporting and executive decision-making.
# MAGIC
# MAGIC The dashboard layer aggregates operational data into high-level KPIs that can be consumed by BI tools such as Power BI.

# COMMAND ----------

df = spark.table(
    "default.wind_turbine_recommendations"
)

display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Executive KPI Summary
# MAGIC
# MAGIC This table provides a high-level overview of turbine performance.
# MAGIC
# MAGIC Key metrics:
# MAGIC
# MAGIC - Average efficiency
# MAGIC - Total energy loss
# MAGIC - Underperformance events
# MAGIC - High risk events

# COMMAND ----------

from pyspark.sql.functions import (
    avg,
    sum,
    count,
    when,
    col
)

dashboard_kpi_summary = df.agg(
    avg("efficiency_ratio").alias("avg_efficiency_ratio"),
    avg("active_power_kw").alias("avg_actual_power_kw"),
    avg("theoretical_power_kwh").alias("avg_theoretical_power_kwh"),
    sum("power_gap_kwh").alias("total_energy_loss_kwh"),
    sum("underperformance_flag").alias("underperformance_events"),
    count("*").alias("total_records")
)

display(dashboard_kpi_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # Monthly Performance Trends
# MAGIC
# MAGIC This table tracks operational performance over time and supports trend analysis.

# COMMAND ----------

dashboard_monthly_performance = (
    df
    .groupBy(
        "year",
        "month"
    )
    .agg(
        avg("active_power_kw").alias("avg_actual_power_kw"),
        avg("theoretical_power_kwh").alias("avg_theoretical_power_kwh"),
        avg("efficiency_ratio").alias("avg_efficiency_ratio"),
        sum("power_gap_kwh").alias("total_energy_loss_kwh")
    )
    .orderBy(
        "year",
        "month"
    )
)

display(
    dashboard_monthly_performance
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Risk Distribution
# MAGIC
# MAGIC This table summarizes operational risk levels and supports maintenance prioritization.

# COMMAND ----------

dashboard_risk_distribution = (
    df
    .groupBy(
        "risk_level"
    )
    .count()
)

display(
    dashboard_risk_distribution
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Wind Category Analysis
# MAGIC
# MAGIC This table evaluates turbine performance under different wind conditions.

# COMMAND ----------

dashboard_wind_analysis = (
    df
    .groupBy(
        "wind_category"
    )
    .agg(
        avg("active_power_kw").alias("avg_actual_power_kw"),
        avg("efficiency_ratio").alias("avg_efficiency_ratio"),
        sum("power_gap_kwh").alias("total_energy_loss_kwh"),
        count("*").alias("records")
    )
)

display(
    dashboard_wind_analysis
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Maintenance Priority Dashboard
# MAGIC
# MAGIC This table identifies the periods with the highest operational losses and should be prioritized for investigation.

# COMMAND ----------

maintenance_priority = (
    df
    .filter(
        col("risk_level") == "High"
    )
    .select(
        "datetime",
        "power_gap_kwh",
        "efficiency_ratio",
        "wind_speed_ms",
        "recommended_action"
    )
    .orderBy(
        col("power_gap_kwh").desc()
    )
)

display(
    maintenance_priority.limit(50)
)

# COMMAND ----------

dashboard_kpi_summary.write.mode("overwrite").saveAsTable(
    "default.dashboard_kpi_summary"
)

dashboard_monthly_performance.write.mode("overwrite").saveAsTable(
    "default.dashboard_monthly_performance"
)

dashboard_risk_distribution.write.mode("overwrite").saveAsTable(
    "default.dashboard_risk_distribution"
)

dashboard_wind_analysis.write.mode("overwrite").saveAsTable(
    "default.dashboard_wind_analysis"
)

maintenance_priority.write.mode("overwrite").saveAsTable(
    "default.dashboard_maintenance_priority"
)