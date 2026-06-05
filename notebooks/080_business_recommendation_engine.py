# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 08 - Business Recommendation Engine
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to convert analytical outputs into actionable business recommendations.
# MAGIC
# MAGIC The recommendation engine uses performance KPIs and prediction-based indicators to identify high-risk operating periods and suggest operational actions.
# MAGIC
# MAGIC This layer helps bridge the gap between machine learning results and business decision-making.

# COMMAND ----------

gold_df = spark.table("default.wind_turbine_gold_advanced")

display(gold_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Recommendation Logic
# MAGIC
# MAGIC The recommendation engine is based on operational rules derived from turbine performance indicators.
# MAGIC
# MAGIC Key indicators:
# MAGIC
# MAGIC - Efficiency Ratio
# MAGIC - Power Gap
# MAGIC - Underperformance Flag
# MAGIC - Wind Speed Category
# MAGIC - Rolling Power Output
# MAGIC
# MAGIC The objective is to classify each record into an operational risk level and assign a recommended action.

# COMMAND ----------

from pyspark.sql.functions import col, when

recommendation_df = gold_df.withColumn(
    "risk_level",
    when(
        (col("efficiency_ratio") < 0.60) &
        (col("power_gap_kwh") > 1000),
        "High"
    )
    .when(
        (col("efficiency_ratio") < 0.80) &
        (col("power_gap_kwh") > 500),
        "Medium"
    )
    .otherwise("Low")
)

display(recommendation_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Recommended Actions
# MAGIC
# MAGIC Each risk level is mapped to a business action.
# MAGIC
# MAGIC Examples:
# MAGIC
# MAGIC - High Risk: Immediate technical inspection
# MAGIC - Medium Risk: Monitor turbine performance
# MAGIC - Low Risk: Normal operation
# MAGIC
# MAGIC This rule-based approach provides interpretable recommendations for operations and maintenance teams.

# COMMAND ----------

recommendation_df = recommendation_df.withColumn(
    "recommended_action",
    when(
        col("risk_level") == "High",
        "Immediate inspection required"
    )
    .when(
        col("risk_level") == "Medium",
        "Monitor performance and schedule review"
    )
    .otherwise("Normal operation")
)

display(
    recommendation_df.select(
        "datetime",
        "active_power_kw",
        "theoretical_power_kwh",
        "power_gap_kwh",
        "efficiency_ratio",
        "risk_level",
        "recommended_action"
    ).limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Business Prioritization
# MAGIC
# MAGIC Operations teams need to prioritize limited maintenance resources.
# MAGIC
# MAGIC The recommendation engine helps identify the most critical periods by ranking observations according to performance loss and risk level.

# COMMAND ----------

high_risk_events = (
    recommendation_df
    .filter(col("risk_level") == "High")
    .select(
        "datetime",
        "active_power_kw",
        "theoretical_power_kwh",
        "power_gap_kwh",
        "efficiency_ratio",
        "wind_speed_ms",
        "wind_category",
        "season",
        "recommended_action"
    )
    .orderBy(col("power_gap_kwh").desc())
)

display(high_risk_events.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC # Risk Distribution
# MAGIC
# MAGIC Risk distribution provides an overview of how frequently the turbine operates under low, medium or high-risk conditions.
# MAGIC
# MAGIC This can support operational reporting and maintenance planning.

# COMMAND ----------

risk_summary = (
    recommendation_df
    .groupBy("risk_level")
    .count()
    .orderBy("risk_level")
)

display(risk_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # Monthly Risk Analysis
# MAGIC
# MAGIC Monthly risk aggregation helps identify whether high-risk events are concentrated during specific periods.
# MAGIC
# MAGIC This can support seasonal maintenance planning and operational scheduling.

# COMMAND ----------

monthly_risk_summary = (
    recommendation_df
    .groupBy("year", "month", "risk_level")
    .count()
    .orderBy("year", "month", "risk_level")
)

display(monthly_risk_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # Estimated Energy Loss
# MAGIC
# MAGIC Estimated energy loss is calculated using the power gap.
# MAGIC
# MAGIC This metric provides a business-oriented view of potential production loss associated with underperformance events.

# COMMAND ----------

from pyspark.sql.functions import sum, avg

loss_summary = (
    recommendation_df
    .groupBy("risk_level")
    .agg(
        sum("power_gap_kwh").alias("total_estimated_energy_loss_kwh"),
        avg("power_gap_kwh").alias("avg_energy_loss_kwh"),
        avg("efficiency_ratio").alias("avg_efficiency_ratio")
    )
    .orderBy("risk_level")
)

display(loss_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Recommendation Layer
# MAGIC
# MAGIC The final recommendation layer stores operational risk levels and recommended actions.
# MAGIC
# MAGIC This table can be used for:
# MAGIC
# MAGIC - Dashboards
# MAGIC - Maintenance prioritization
# MAGIC - Operational monitoring
# MAGIC - Business reporting

# COMMAND ----------

recommendation_df.write.mode("overwrite").saveAsTable(
    "default.wind_turbine_recommendations"
)

print("Recommendation table created successfully.")

# COMMAND ----------

final_df = spark.table("default.wind_turbine_recommendations")

display(final_df.limit(10))

print("Rows:", final_df.count())
print("Columns:", len(final_df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC # Recommendation Engine Summary
# MAGIC
# MAGIC Completed Activities:
# MAGIC
# MAGIC - Created operational risk levels
# MAGIC - Mapped risk levels to recommended business actions
# MAGIC - Identified high-risk events
# MAGIC - Calculated risk distribution
# MAGIC - Estimated energy loss by risk level
# MAGIC - Saved final recommendation table
# MAGIC
# MAGIC Output Asset:
# MAGIC
# MAGIC default.wind_turbine_recommendations
# MAGIC
# MAGIC Business Value:
# MAGIC
# MAGIC The recommendation engine converts turbine performance analytics into actionable maintenance and operations insights.