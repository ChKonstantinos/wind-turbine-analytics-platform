# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 10 - Predictive Maintenance Scoring
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC The purpose of this notebook is to transform operational analytics into a maintenance-oriented health scoring system.
# MAGIC
# MAGIC The Health Score provides a single indicator that summarizes turbine performance and operational condition.
# MAGIC
# MAGIC The score can be used for:
# MAGIC
# MAGIC - Maintenance prioritization
# MAGIC - Operational monitoring
# MAGIC - Executive reporting
# MAGIC - Asset health management

# COMMAND ----------

df = spark.table(
    "default.wind_turbine_recommendations"
)

display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC # Health Score Logic
# MAGIC
# MAGIC The Health Score is derived from:
# MAGIC
# MAGIC - Efficiency Ratio
# MAGIC - Power Gap
# MAGIC - Underperformance Events
# MAGIC
# MAGIC Higher scores indicate healthier operation.
# MAGIC
# MAGIC Lower scores indicate potential maintenance needs.

# COMMAND ----------

from pyspark.sql.functions import (
    col,
    least,
    greatest,
    lit
)

health_df = df.withColumn(
    "health_score",
    least(
        greatest(
            (
                col("efficiency_ratio") * 100
            ) - (
                col("power_gap_kwh") / 50
            ),
            lit(0)
        ),
        lit(100)
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Health Categories
# MAGIC
# MAGIC The health score is translated into operational categories.
# MAGIC
# MAGIC This simplifies maintenance planning and communication with non-technical stakeholders.

# COMMAND ----------

from pyspark.sql.functions import when

health_df = health_df.withColumn(
    "health_status",
    when(
        col("health_score") >= 90,
        "Healthy"
    )
    .when(
        col("health_score") >= 75,
        "Warning"
    )
    .otherwise(
        "Critical"
    )
)

# COMMAND ----------

display(
    health_df.select(
        "datetime",
        "health_score",
        "health_status",
        "risk_level"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Health Score Distribution
# MAGIC
# MAGIC This analysis shows how frequently the turbine operates under each health condition.

# COMMAND ----------

health_distribution = (
    health_df
    .groupBy(
        "health_status"
    )
    .count()
)

display(
    health_distribution
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Monthly Health Monitoring
# MAGIC
# MAGIC Health trends over time can reveal gradual performance degradation and support preventive maintenance planning.

# COMMAND ----------

from pyspark.sql.functions import avg

monthly_health = (
    health_df
    .groupBy(
        "year",
        "month"
    )
    .agg(
        avg("health_score")
        .alias("avg_health_score")
    )
    .orderBy(
        "year",
        "month"
    )
)

display(
    monthly_health
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Maintenance Priority Ranking
# MAGIC
# MAGIC The lowest health score observations represent the highest maintenance priority.

# COMMAND ----------

maintenance_priority = (
    health_df
    .select(
        "datetime",
        "health_score",
        "health_status",
        "power_gap_kwh",
        "efficiency_ratio",
        "recommended_action"
    )
    .orderBy(
        col("health_score")
    )
)

display(
    maintenance_priority.limit(50)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Maintenance Layer
# MAGIC
# MAGIC The maintenance layer contains health indicators and prioritization metrics that can be consumed by dashboards and maintenance planning systems.

# COMMAND ----------

health_df.write.mode("overwrite").saveAsTable(
    "default.wind_turbine_health"
)

print("Health table created successfully.")

# COMMAND ----------

final_health = spark.table(
    "default.wind_turbine_health"
)

display(
    final_health.limit(10)
)