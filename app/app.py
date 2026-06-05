import streamlit as st
import plotly.express as px

from databricks_connection import run_query


st.set_page_config(
    page_title="Wind Turbine Analytics Platform",
    page_icon="🌬️",
    layout="wide"
)


@st.cache_data(ttl=600)
def load_table(query: str):
    return run_query(query)


def load_data():
    kpi_df = load_table("SELECT * FROM default.dashboard_kpi_summary")
    monthly_df = load_table("SELECT * FROM default.dashboard_monthly_performance")
    risk_df = load_table("SELECT * FROM default.dashboard_risk_distribution")
    wind_df = load_table("SELECT * FROM default.dashboard_wind_analysis")
    maintenance_df = load_table("""
        SELECT *
        FROM default.dashboard_maintenance_priority
        LIMIT 200
    """)
    health_df = load_table("""
        SELECT health_status, COUNT(*) AS count
        FROM default.wind_turbine_health
        GROUP BY health_status
    """)
    return kpi_df, monthly_df, risk_df, wind_df, maintenance_df, health_df


st.sidebar.title("🌬️ Wind Turbine Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Performance Analysis",
        "Risk Analysis",
        "Maintenance Center",
        "Health Monitoring"
    ]
)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()


kpi_df, monthly_df, risk_df, wind_df, maintenance_df, health_df = load_data()

st.title("Wind Turbine Analytics & Predictive Maintenance Platform")
st.caption("Data source: Databricks SQL Warehouse")


if page == "Executive Overview":
    st.header("Executive Overview")

    kpi = kpi_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Efficiency",
        f"{kpi['avg_efficiency_ratio'] * 100:.1f}%"
    )

    col2.metric(
        "Average Actual Power",
        f"{kpi['avg_actual_power_kw']:.0f} kW"
    )

    col3.metric(
        "Total Energy Loss",
        f"{kpi['total_energy_loss_kwh']:,.0f} kWh"
    )

    col4.metric(
        "Underperformance Events",
        f"{int(kpi['underperformance_events']):,}"
    )

    st.markdown("### Monthly Efficiency Trend")

    monthly_df["period"] = (
        monthly_df["year"].astype(str)
        + "-"
        + monthly_df["month"].astype(str).str.zfill(2)
    )

    fig = px.line(
        monthly_df,
        x="period",
        y="avg_efficiency_ratio",
        markers=True,
        title="Average Efficiency Ratio by Month"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Performance Analysis":
    st.header("Performance Analysis")

    monthly_df["period"] = (
        monthly_df["year"].astype(str)
        + "-"
        + monthly_df["month"].astype(str).str.zfill(2)
    )

    fig_loss = px.bar(
        monthly_df,
        x="period",
        y="total_energy_loss_kwh",
        title="Total Energy Loss by Month",
        text="total_energy_loss_kwh"
    )

    st.plotly_chart(fig_loss, use_container_width=True)

    fig_wind = px.bar(
        wind_df,
        x="wind_category",
        y="avg_efficiency_ratio",
        title="Average Efficiency by Wind Category",
        text="avg_efficiency_ratio"
    )

    st.plotly_chart(fig_wind, use_container_width=True)

    st.dataframe(wind_df, use_container_width=True)


elif page == "Risk Analysis":
    st.header("Risk Analysis")

    fig_risk = px.bar(
        risk_df,
        x="risk_level",
        y="count",
        title="Operational Risk Distribution",
        text="count"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    st.dataframe(risk_df, use_container_width=True)


elif page == "Maintenance Center":
    st.header("Maintenance Center")

    st.markdown("""
    This section shows the highest-priority operating periods based on estimated energy loss.
    These records can be used by maintenance teams to prioritize inspections.
    """)

    st.dataframe(maintenance_df, use_container_width=True)

    csv = maintenance_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Maintenance Report CSV",
        data=csv,
        file_name="maintenance_priority_report.csv",
        mime="text/csv"
    )


elif page == "Health Monitoring":
    st.header("Health Monitoring")

    fig_health = px.pie(
        health_df,
        names="health_status",
        values="count",
        title="Turbine Health Status Distribution"
    )

    st.plotly_chart(fig_health, use_container_width=True)

    st.dataframe(health_df, use_container_width=True)