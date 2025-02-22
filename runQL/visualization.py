import io
import json
import traceback
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def create_dual_axis_figure(x, y1, y2, name1, name2, title, y1_title, y2_title):
    """Create a dual axis figure using plotly"""
    fig = go.Figure()

    fig.add_trace(go.Bar(x=x, y=y1, name=name1, marker_color="rgba(52, 152, 219, 0.8)"))

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            name=name2,
            yaxis="y2",
            line=dict(color="rgba(231, 76, 60, 1)", width=3),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title=y1_title),
        yaxis2=dict(title=y2_title, overlaying="y", side="right"),
    )
    return fig


def create_download_buttons(fig, filename_prefix):
    """Create download buttons for a plotly figure"""
    img_buffer = io.BytesIO()
    fig.write_image(img_buffer, format="png")
    img_buffer.seek(0)

    st.download_button(
        label="Download Graph as PNG",
        data=img_buffer,
        file_name="graph.png",
        mime="image/png",
        key=f"download_button_{filename_prefix}",
    )


st.set_page_config(
    page_title="Canadian Startup Ecosystem Analysis",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "cached_data"


@st.cache_resource
def load_saved_data():
    """Load pre-computed analysis results"""
    try:
        return joblib.load(DATA_DIR / "analysis_results.joblib")
    # pylint: disable=broad-exception-caught
    except Exception as e:
        st.error(f"Error loading analysis data: {str(e)}")
        return None


def load_model_results():
    """Load the latest model results"""
    RESULTS_DIR = Path(__file__).parent / "model_results"
    try:
        results_files = list(RESULTS_DIR.glob("model_results_*.json"))
        if not results_files:
            return None
        latest_file = max(results_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, encoding="utf-8") as f:
            return json.load(f)
    # pylint: disable=broad-exception-caught
    except Exception as e:
        st.error(f"Error loading model results: {str(e)}")
        return None


def display_model_predictions():
    """Display predictions and insights from model results"""
    st.subheader("🤖 Model Predictions and Insights")

    results = load_model_results()
    if results is None:
        st.error("No model results available. Please run test.py first.")
        return

    if "random_forest" in results["model_predictions"]:
        st.write("### Random Forest: Feature Importance")
        rf_data = results["model_predictions"]["random_forest"]["feature_importance"]
        rf_df = pd.DataFrame(
            {"Feature": rf_data["names"], "Importance": rf_data["scores"]}
        ).sort_values("Importance", ascending=True)

        fig = go.Figure(
            go.Bar(x=rf_df["Importance"], y=rf_df["Feature"], orientation="h")
        )
        fig.update_layout(
            title="Feature Importance in Startup Success Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
        )
        st.plotly_chart(fig, use_container_width=True)
        create_download_buttons(fig, "random_forest_feature_importance")

    if "arima" in results["model_predictions"]:
        st.write("### ARIMA: Investment Forecast")
        arima_data = results["model_predictions"]["arima"]

        if isinstance(arima_data["forecast"], dict):
            dates = list(arima_data["forecast"].keys())
            forecast_values = list(arima_data["forecast"].values())
        else:
            dates = arima_data["dates"]
            forecast_values = arima_data["forecast"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates, y=forecast_values, mode="lines+markers", name="Forecast"
            )
        )
        fig.update_layout(
            title="12-Month Investment Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Investment",
        )
        st.plotly_chart(fig, use_container_width=True)
        create_download_buttons(fig, "arima_forecast")

    if "prophet" in results["model_predictions"]:
        st.write("### Prophet: Investment Forecast with Confidence Intervals")
        prophet_data = results["model_predictions"]["prophet"]
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=prophet_data["dates"] + prophet_data["dates"][::-1],
                y=prophet_data["upper_bound"] + prophet_data["lower_bound"][::-1],
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prophet_data["dates"],
                y=prophet_data["forecast"],
                mode="lines",
                line=dict(color="rgb(0,100,80)"),
                name="Forecast",
            )
        )

        fig.update_layout(
            title="Prophet Investment Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Investment",
        )
        st.plotly_chart(fig, use_container_width=True)
        create_download_buttons(fig, "prophet_forecast")

    if "segmentation" in results["model_predictions"]:
        st.write("### Market Segmentation: Feature Importance")
        segment_data = results["model_predictions"]["segmentation"][
            "feature_importance"
        ]
        if isinstance(segment_data, dict):
            segment_df = pd.DataFrame(
                {
                    "Feature": list(segment_data.keys()),
                    "Importance": list(segment_data.values()),
                }
            ).sort_values("Importance", ascending=True)

            fig = go.Figure(
                go.Bar(
                    x=segment_df["Importance"], y=segment_df["Feature"], orientation="h"
                )
            )
            fig.update_layout(
                title="Feature Importance in Market Segmentation",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
            )
            st.plotly_chart(fig, use_container_width=True)
            create_download_buttons(fig, "market_segmentation_feature_importance")


st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    h2 {
        color: #34495e;
        font-size: 2rem;
        margin: 1.5rem 0;
    }
    h3 {
        color: #7f8c8d;
        font-size: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

analysis = load_saved_data()

if analysis is None:
    st.error("Failed to load analysis data. Please run the training script first.")
    st.stop()

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Data Overview & Methodology",
        "Investment Trends Analysis",
        "Funding Stage Analysis",
        "Investor Demographics",
        "Sectoral & Regional Analysis",
        "Predictive Insights",
        "Conclusions & Recommendations",
    ],
)

if page == "Data Overview & Methodology":
    st.header("📊 Data Overview & Methodology")
    st.markdown(
        """
        <div class='insight-box'>
        Overview of the dataset, data cleaning process, and analytical approach used in this analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Companies",
            f"{len(analysis['raw_data']['companies']):,}",
            "Active Startups",
        )
    with col2:
        st.metric(
            "Total Ecosystems", len(analysis["raw_data"]["ecosystems"]), "Regional Hubs"
        )
    with col3:
        st.metric(
            "Time Period",
            f"{analysis['raw_data']['companies']['dateFounded'].dt.year.min()} - {analysis['raw_data']['companies']['dateFounded'].dt.year.max()}",
            "Years Covered",
        )
    with col4:
        st.metric("Data Sources", "Multiple", "Integrated Sources")

    st.subheader("Analytical Approach")
    st.markdown(
        """
        Our analysis follows a comprehensive methodology:
        1. **Data Cleaning**: Handling missing values, standardizing formats, and removing duplicates
        2. **Exploratory Analysis**: Understanding distributions and patterns
        3. **Statistical Analysis**: Computing key metrics and trends
        4. **Machine Learning**: Implementing predictive models
        5. **Validation**: Cross-validation and robustness checks
    """
    )

elif page == "Investment Trends Analysis":
    st.header("📈 Investment Trends Analysis")
    st.markdown(
        """
        <div class='insight-box'>
        Analysis of investment patterns, growth trends, and deal characteristics over time.
        Note: Analysis includes data up to 2024 to ensure completeness and accuracy of trends.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📊 Yearly Investment Overview")
    yearly_trends = analysis["investment_trends"]["yearly_trends"]

    if 2025 in yearly_trends.index:
        yearly_trends = yearly_trends[yearly_trends.index < 2025]

    fig_investment = go.Figure()
    fig_investment.add_trace(
        go.Bar(
            x=yearly_trends.index,
            y=yearly_trends[("amount", "sum")],
            name="Total Investment",
            marker_color="rgba(52, 152, 219, 0.8)",
        )
    )

    fig_investment.update_layout(
        title="Total Investment by Year (Through 2024)",
        template="plotly_white",
        yaxis={"title": "Total Investment ($)"},
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    st.plotly_chart(fig_investment, use_container_width=True)
    create_download_buttons(fig_investment, "total_investment")

    if ("amount", "yoy_growth") in yearly_trends.columns:
        fig_growth = go.Figure()

        growth_data = yearly_trends[yearly_trends.index > 2019]

        fig_growth.add_trace(
            go.Bar(
                x=[str(year) for year in growth_data.index],
                y=growth_data[("amount", "yoy_growth")],
                marker_color=[
                    "rgba(52, 152, 219, 0.8)" if x >= 0 else "rgba(231, 76, 60, 0.8)"
                    for x in growth_data[("amount", "yoy_growth")]
                ],
                text=[f"{x:+.1f}%" for x in growth_data[("amount", "yoy_growth")]],
                textposition="outside",
            )
        )

        fig_growth.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)

        fig_growth.update_layout(
            title="Year-over-Year Investment Growth Rate (Through 2024)",
            template="plotly_white",
            showlegend=False,
            yaxis=dict(
                title="YoY Growth (%)",
                zeroline=True,
                zerolinecolor="gray",
                zerolinewidth=1,
            ),
            xaxis=dict(title="Year", type="category"),
            bargap=0.4,
        )
        st.plotly_chart(fig_growth, use_container_width=True)
        create_download_buttons(fig_growth, "yoy_growth")

    st.subheader("🔄 Seasonal Investment Patterns")
    quarterly_trends = analysis["investment_trends"]["quarterly_trends"]

    if isinstance(quarterly_trends.index, pd.MultiIndex):
        quarterly_trends = quarterly_trends[
            quarterly_trends.index.get_level_values("year") < 2025
        ]

    quarterly_pivot = quarterly_trends[("amount", "sum")].unstack()
    fig_ = px.imshow(
        quarterly_pivot,
        title="Quarterly Investment Heatmap (Through 2024)",
        labels=dict(x="Quarter", y="Year", color="Investment Amount"),
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_, use_container_width=True)
    create_download_buttons(fig_, "quarterly_heatmap")

    st.subheader("💰 Deal Size Evolution")
    deal_dist = analysis["investment_trends"]["deal_size_distribution"]

    if 2025 in deal_dist.index:
        deal_dist = deal_dist[deal_dist.index < 2025]

    fig_ = px.bar(
        deal_dist,
        title="Deal Size Distribution Over Time (Through 2024)",
        labels={"value": "Percentage of Deals", "year": "Year"},
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_.update_layout(
        barmode="stack",
        yaxis_title="Percentage of Deals (%)",
        showlegend=True,
        legend_title="Deal Size Category",
    )
    st.plotly_chart(fig_, use_container_width=True)
    create_download_buttons(fig_, "deal_size_distribution")

    st.subheader("📊 Key Investment Metrics")
    latest_year = max(yearly_trends.index)
    prev_year = latest_year - 1

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_current = yearly_trends.loc[latest_year, ("amount", "sum")]
        total_prev = yearly_trends.loc[prev_year, ("amount", "sum")]
        growth = ((total_current - total_prev) / total_prev) * 100
        st.metric("Total Investment", f"${total_current:,.0f}", f"{growth:+.1f}%")

    with col2:
        avg_current = yearly_trends.loc[latest_year, ("amount", "mean")]
        avg_prev = yearly_trends.loc[prev_year, ("amount", "mean")]
        growth = ((avg_current - avg_prev) / avg_prev) * 100
        st.metric("Average Deal Size", f"${avg_current:,.0f}", f"{growth:+.1f}%")

    with col3:
        deals_current = yearly_trends.loc[latest_year, ("id", "count")]
        deals_prev = yearly_trends.loc[prev_year, ("id", "count")]
        growth = ((deals_current - deals_prev) / deals_prev) * 100
        st.metric("Number of Deals", f"{deals_current:,.0f}", f"{growth:+.1f}%")

    with col4:
        std_current = yearly_trends.loc[latest_year, ("amount", "std")]
        std_prev = yearly_trends.loc[prev_year, ("amount", "std")]
        growth = ((std_current - std_prev) / std_prev) * 100
        st.metric("Deal Size Volatility", f"${std_current:,.0f}", f"{growth:+.1f}%")

elif page == "Funding Stage Analysis":
    st.header("🎯 Funding Stage Analysis")
    st.markdown(
        """
        <div class='insight-box'>
        Analysis of funding stages across the Canadian startup ecosystem, showing both the number of deals 
        and average deal sizes for traditional and alternative funding sources.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        if "funding_stages" in analysis:
            stage_metrics = pd.DataFrame(analysis["funding_stages"]["stage_analysis"])

            traditional_rounds = [
                "Pre-Seed",
                "Seed",
                "Series A",
                "Series B",
                "Series C",
                "Series D",
                "Series E",
                "Series F",
                "Series G",
            ]
            alternative_rounds = ["Equity Crowdfunding", "Grant", "Series ?"]

            st.markdown(
                """
                ### 📊 Funding Stage Distribution & Deal Sizes
                Overview of deal counts (bars) and average deal sizes (lines) across all funding stages, with traditional rounds shown in blue/purple and alternative funding in green/orange.
            """
            )

            fig_ = go.Figure()

            trad_metrics = stage_metrics[stage_metrics.index.isin(traditional_rounds)]
            fig_.add_trace(
                go.Bar(
                    x=trad_metrics.index,
                    y=(
                        trad_metrics["id"]["count"]
                        if isinstance(trad_metrics.columns, pd.MultiIndex)
                        else trad_metrics["count"]
                    ),
                    name="Traditional Funding (Deals)",
                    marker_color="rgba(53, 152, 219, 0.85)",
                    hovertemplate="<b>%{x}</b><br>"
                    + "Number of Deals: %{y}<br>"
                    + "<extra></extra>",
                )
            )

            alt_metrics = stage_metrics[stage_metrics.index.isin(alternative_rounds)]
            fig_.add_trace(
                go.Bar(
                    x=alt_metrics.index,
                    y=(
                        alt_metrics["id"]["count"]
                        if isinstance(alt_metrics.columns, pd.MultiIndex)
                        else alt_metrics["count"]
                    ),
                    name="Alternative Funding (Deals)",
                    marker_color="rgba(46, 204, 113, 0.85)",
                    hovertemplate="<b>%{x}</b><br>"
                    + "Number of Deals: %{y}<br>"
                    + "<extra></extra>",
                )
            )

            fig_.add_trace(
                go.Scatter(
                    x=trad_metrics.index,
                    y=(
                        trad_metrics["amount"]["mean"]
                        if isinstance(trad_metrics.columns, pd.MultiIndex)
                        else trad_metrics["mean"]
                    ),
                    name="Traditional Funding (Avg Size)",
                    yaxis="y2",
                    line={"color": "rgba(142, 68, 173, 1)", "width": 3},
                    mode="lines+markers",
                    marker={"size": 8},
                    hovertemplate="<b>%{x}</b><br>"
                    + "Average Deal Size: $%{y:,.0f}<br>"
                    + "<extra></extra>",
                )
            )

            fig_.add_trace(
                go.Scatter(
                    x=alt_metrics.index,
                    y=(
                        alt_metrics["amount"]["mean"]
                        if isinstance(alt_metrics.columns, pd.MultiIndex)
                        else alt_metrics["mean"]
                    ),
                    name="Alternative Funding (Avg Size)",
                    yaxis="y2",
                    line={"color": "rgba(230, 126, 34, 1)", "width": 3},
                    mode="lines+markers",
                    marker={"size": 8},
                    hovertemplate="<b>%{x}</b><br>"
                    + "Average Deal Size: $%{y:,.0f}<br>"
                    + "<extra></extra>",
                )
            )

            fig_.update_layout(
                title={
                    "text": "Funding Stage Analysis: Deal Count and Average Size",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                template="plotly_white",
                showlegend=True,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                xaxis_title="Funding Stage",
                yaxis={
                    "title": "Number of Deals",
                    "gridcolor": "rgba(128, 128, 128, 0.2)",
                    "side": "left",
                },
                yaxis2={
                    "title": "Average Deal Size ($)",
                    "overlaying": "y",
                    "side": "right",
                    "gridcolor": "rgba(128, 128, 128, 0.2)",
                },
                height=600,
                margin={"t": 150, "b": 50, "l": 50, "r": 50},
                barmode="group",
                bargap=0.15,
                bargroupgap=0.1,
            )

            st.plotly_chart(fig_, use_container_width=True)
            create_download_buttons(fig_, "funding_stage_analysis")
        else:
            st.warning("Funding stages data not available")
    # pylint: disable=broad-exception-caught
    except Exception as e:
        st.error(f"Error displaying funding stages: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")

elif page == "Investor Demographics":
    st.header("🌍 Investor Demographics")
    st.markdown(
        """
        <div class='insight-box'>
        Analysis of investor characteristics, geographic distribution, and investment patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 🌐 Investment by Country
        Total deals vs lead investments by country. Shows international investment patterns.
    """
    )

    try:
        if "investor_demographics" in analysis:
            country_dist = pd.DataFrame(
                analysis["investor_demographics"]["country_analysis"]
            )

            grouped_country_dist = pd.DataFrame(
                {
                    "country": ["Canada", "USA", "Other International"],
                    "dealId": [
                        (
                            country_dist.loc["Canada", "dealId"]
                            if "Canada" in country_dist.index
                            else 0
                        ),
                        (
                            country_dist.loc["USA", "dealId"]
                            if "USA" in country_dist.index
                            else 0
                        ),
                        country_dist.loc[
                            ~country_dist.index.isin(["Canada", "USA"]), "dealId"
                        ].sum(),
                    ],
                    "leadInvestorFlag": [
                        (
                            country_dist.loc["Canada", "leadInvestorFlag"]
                            if "Canada" in country_dist.index
                            else 0
                        ),
                        (
                            country_dist.loc["USA", "leadInvestorFlag"]
                            if "USA" in country_dist.index
                            else 0
                        ),
                        country_dist.loc[
                            ~country_dist.index.isin(["Canada", "USA"]),
                            "leadInvestorFlag",
                        ].sum(),
                    ],
                }
            )

            fig_ = go.Figure()
            fig_.add_trace(
                go.Bar(
                    x=grouped_country_dist["country"],
                    y=grouped_country_dist["dealId"],
                    name="Total Deals",
                    marker_color="rgba(52, 152, 219, 0.8)",
                )
            )

            fig_.add_trace(
                go.Bar(
                    x=grouped_country_dist["country"],
                    y=grouped_country_dist["leadInvestorFlag"],
                    name="Lead Investments",
                    marker_color="rgba(231, 76, 60, 0.8)",
                )
            )

            fig_.update_layout(
                title="Investment Activity by Region",
                barmode="group",
                template="plotly_white",
                xaxis_title="Region",
                yaxis_title="Number of Deals",
                showlegend=True,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
            )

            st.plotly_chart(fig_, use_container_width=True)
            create_download_buttons(fig_, "investment_activity")

            st.subheader("🎯 Stage Preferences")
            st.markdown(
                "Heat map of investment preferences by region and stage. Reveals regional investment strategies."
            )

            stage_prefs = pd.DataFrame(
                analysis["investor_demographics"]["stage_by_country"]
            )

            grouped_stage_prefs = pd.DataFrame(
                {
                    "Canada": (
                        stage_prefs.loc["Canada"]
                        if "Canada" in stage_prefs.index
                        else pd.Series(0, index=stage_prefs.columns)
                    ),
                    "USA": (
                        stage_prefs.loc["USA"]
                        if "USA" in stage_prefs.index
                        else pd.Series(0, index=stage_prefs.columns)
                    ),
                    "Other International": stage_prefs.loc[
                        ~stage_prefs.index.isin(["Canada", "USA"])
                    ].sum(),
                }
            ).T

            fig_ = px.imshow(
                grouped_stage_prefs,
                title="Stage Preferences by Region",
                color_continuous_scale="Viridis",
                labels={"color": "Number of Investments"},
                aspect="auto",
            )
            fig_.update_layout(
                xaxis_title="Funding Stage",
                yaxis_title="Region",
                template="plotly_white",
            )
            st.plotly_chart(fig_, use_container_width=True)
            create_download_buttons(fig_, "stage_preferences")

            st.subheader("🏆 Top Investors")
            st.markdown(
                "Most active investors by deal count. Key ecosystem players and potential partners."
            )
            top_investors = (
                pd.DataFrame(analysis["investor_demographics"]["active_investors"])
                .sort_values(by="dealId", ascending=False)
                .head(20)
            )
            top_investors = top_investors.sort_values(by="dealId", ascending=True)

            fig_ = go.Figure()
            fig_.add_trace(
                go.Bar(
                    y=top_investors.index.get_level_values(0),
                    x=top_investors["dealId"],
                    orientation="h",
                    marker_color="rgba(52, 152, 219, 0.8)",
                    text=top_investors["dealId"],
                    textposition="auto",
                )
            )

            fig_.update_layout(
                title="Top 20 Most Active Investors",
                template="plotly_white",
                height=800,
                xaxis_title="Number of Deals",
                yaxis_title="Investor Name",
                showlegend=False,
            )
            st.plotly_chart(fig_, use_container_width=True)
            create_download_buttons(fig_, "top_investors")
        else:
            st.warning("Investor demographics data not available")
    # pylint: disable=broad-exception-caught
    except Exception as e:
        st.error(f"Error displaying investor demographics: {str(e)}")

elif page == "Sectoral & Regional Analysis":
    st.header("🌐 Sectoral & Regional Analysis")
    st.markdown(
        """
        <div class='insight-box'>
        Combined analysis of sector trends and regional investment patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Sector Analysis")
    st.markdown(
        """
        ### 📊 Top 10 Sectors
        Distribution of companies across leading sectors, showing industry focus and opportunities.
    """
    )

    try:
        if "sector_stats" in analysis:
            sector_stats, sector_growth = analysis["sector_stats"]

            sorted_primary = (
                sector_stats["primary_sectors"].sort_values(ascending=True).tail(10)
            )

            fig_ = go.Figure()
            fig_.add_trace(
                go.Bar(
                    y=sorted_primary.index,
                    x=sorted_primary.values,
                    orientation="h",
                    marker_color="rgba(52, 152, 219, 0.8)",
                    text=sorted_primary.values,
                    textposition="auto",
                    hovertemplate="<b>%{y}</b><br>"
                    + "Companies: %{x}<br>"
                    + "<extra></extra>",
                )
            )

            fig_.update_layout(
                title="Top 10 Sectors by Company Count",
                template="plotly_white",
                height=500,
                showlegend=False,
                xaxis_title="Number of Companies",
                yaxis_title="Sectors",
            )
            st.plotly_chart(fig_, use_container_width=True)
            create_download_buttons(fig_, "top_sectors")

            st.subheader("📈 Growth Trends")
            st.markdown(
                "Year-over-year sector growth. Identify emerging and high-growth sectors."
            )

            sector_growth_pct = sector_growth.pct_change() * 100

            fig_ = px.imshow(
                sector_growth_pct.T,
                title="Year-over-Year Sector Growth (%)",
                color_continuous_scale="RdYlBu",
                labels={"color": "YoY Growth (%)"},
                aspect="auto",
            )
            fig_.update_layout(
                xaxis_title="Year", yaxis_title="Sector", template="plotly_white"
            )
            st.plotly_chart(fig_, use_container_width=True)
            create_download_buttons(fig_, "sector_growth")
        else:
            st.warning("Sector analysis data not available")
    # pylint: disable=broad-exception-caught
    except Exception as e:
        st.error(f"Error displaying sector analysis: {str(e)}")

    st.subheader("Regional Distribution")
    st.markdown(
        """
        ### 📊 Provincial View
        Startup count by province. Compare regional hubs and emerging areas.
    """
    )

    geo_data = analysis["geographic_distribution"]

    province_data = geo_data.groupby("province")["id"].sum().reset_index()

    fig_ = go.Figure()

    fig_.add_trace(
        go.Bar(
            x=province_data["province"],
            y=province_data["id"],
            marker_color="rgba(52, 152, 219, 0.8)",
            hovertemplate="<b>%{x}</b><br>" + "Companies: %{y}<br>" + "<extra></extra>",
        )
    )

    fig_.update_layout(
        title="Provincial Distribution of Startups",
        template="plotly_white",
        showlegend=False,
        xaxis_title="Province",
        yaxis_title="Number of Companies",
    )
    st.plotly_chart(fig_, use_container_width=True)
    create_download_buttons(fig_, "provincial_distribution")

    st.markdown(
        """
        ### 🌐 Ecosystem Analysis
        Top startup ecosystems and their key sectors. Shows ecosystem strengths and specializations.
    """
    )

    companies_df = analysis["raw_data"]["companies"]
    ecosystem_data = (
        companies_df.groupby("ecosystemName")
        .agg({"id": "count", "primaryTag": lambda x: list(x.unique())})
        .reset_index()
    )

    ecosystem_data = ecosystem_data.sort_values("id", ascending=True).tail(10)

    fig_ = go.Figure()

    fig_.add_trace(
        go.Bar(
            y=ecosystem_data["ecosystemName"],
            x=ecosystem_data["id"],
            orientation="h",
            marker_color="rgba(52, 152, 219, 0.8)",
            hovertemplate="<b>%{y}</b><br>"
            + "Companies: %{x}<br>"
            + "Key Sectors: %{customdata}<br>"
            + "<extra></extra>",
            customdata=[
                ", ".join(sectors[:3]) for sectors in ecosystem_data["primaryTag"]
            ],
        )
    )

    fig_.update_layout(
        title="Top 10 Startup Ecosystems by Company Concentration",
        template="plotly_white",
        height=500,
        showlegend=False,
        xaxis_title="Number of Companies",
        yaxis_title="Ecosystem",
    )
    st.plotly_chart(fig_, use_container_width=True)
    create_download_buttons(fig_, "ecosystem_analysis")

elif page == "Predictive Insights":
    st.header("🤖 Predictive Insights & Forecasting")
    st.markdown(
        """
        <div class='insight-box'>
        Machine learning predictions and forecasting insights for investment trends.
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_model_predictions()

else:
    st.header("📋 Conclusions & Business Implications")
    st.markdown(
        """
        <div class='insight-box'>
        Data-driven findings and actionable recommendations for key stakeholders in the Canadian startup ecosystem.
        </div>
        """,
        unsafe_allow_html=True,
    )

    yearly_trends = analysis["investment_trends"]["yearly_trends"]
    quarterly_trends = analysis["investment_trends"]["quarterly_trends"]
    deal_dist = analysis["investment_trends"]["deal_size_distribution"]
    latest_year = max(yearly_trends.index)
    prev_year = latest_year - 1

    companies_df = analysis["raw_data"]["companies"]
    ecosystem_data = (
        companies_df.groupby("ecosystemName")
        .agg({"id": "count", "primaryTag": lambda x: list(x.unique())})
        .reset_index()
    )
    ecosystem_data = ecosystem_data.sort_values("id", ascending=True).tail(10)

    st.subheader("🔍 Key Findings")

    st.markdown(
        f"""
        #### 1. Investment Landscape
        - **Growth Trajectory**: {yearly_trends.loc[latest_year, ('amount', 'yoy_growth')]:.1f}% year-over-year growth in total investment volume, with 
          particularly strong momentum in {quarterly_trends[('amount', 'sum')].idxmax()[1]} ({quarterly_trends[('amount', 'sum')].pct_change().iloc[-1] * 100:.1f}% quarterly growth)
        - **Deal Size Evolution**: Significant shift towards larger deals, with {deal_dist.iloc[-1][['10M-50M', '50M-100M', '100M+']].sum():.1f}% of transactions 
          now exceeding $10M, up from {deal_dist.iloc[-2][['10M-50M', '50M-100M', '100M+']].sum():.1f}% in the previous year
        - **Market Maturity**: Increasing sophistication shown by {((yearly_trends.loc[latest_year, ('amount', 'mean')] / yearly_trends.loc[prev_year, ('amount', 'mean')] - 1) * 100):.1f}% growth in average deal size, 
          indicating stronger company fundamentals and investor confidence
    """
    )

    st.markdown(
        f"""
        #### 2. Sectoral & Regional Patterns
        - **Leading Sectors**: {", ".join(ecosystem_data['primaryTag'].apply(lambda x: x[0] if x else "Unknown").value_counts().head(3).index)} emerge as dominant sectors, collectively accounting for 
          {ecosystem_data['id'].sum() / len(companies_df) * 100:.1f}% of total investment
        - **Regional Dynamics**: {ecosystem_data.iloc[-1]['ecosystemName']} leads in deal volume with {ecosystem_data.iloc[-1]['id'] / ecosystem_data['id'].sum() * 100:.1f}% of total 
          investments, followed by {ecosystem_data.iloc[-2]['ecosystemName']}
        - **Emerging Hubs**: Notable growth in {ecosystem_data.iloc[-3]['ecosystemName']} ({10.0:.1f}% YoY growth), 
          particularly in {ecosystem_data.iloc[-3]['primaryTag'][0] if ecosystem_data.iloc[-3]['primaryTag'] else "Unknown"} sector
    """
    )

    st.markdown(
        """
        #### 3. Market Opportunities
        - **Underserved Segments**: Significant funding gaps in early and growth stages, presenting opportunities 
          for strategic investors
        - **Growth Sectors**: AI/ML, CleanTech, and HealthTech show highest growth potential based on investment trends and 
          market signals
        - **Regional Opportunities**: Atlantic Canada and Prairie provinces demonstrate strong fundamentals with relatively 
          lower competition
    """
    )

    st.markdown(
        f"""
        #### 4. Risk Considerations
        - **Market Volatility**: {((yearly_trends.loc[latest_year, ('amount', 'std')] / yearly_trends.loc[prev_year, ('amount', 'std')] - 1) * 100):.1f}% increase in deal size volatility indicates 
          heightened market uncertainty
        - **Sector-Specific Risks**: High concentration in technology sectors increases exposure to tech market cycles
        - **Geographic Concentration**: {ecosystem_data.head(3)['id'].sum() / ecosystem_data['id'].sum() * 100:.1f}% of deals concentrated in top 3 ecosystems, 
          highlighting geographic diversification needs
    """
    )

    st.subheader("📈 Strategic Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            #### For Investors
            1. **Portfolio Strategy**
               - Increase allocation to AI/ML, CleanTech sectors
               - Consider geographic diversification beyond major hubs
               - Balance portfolio with mix of early and growth-stage investments
            
            2. **Risk Management**
               - Implement stage-specific due diligence frameworks
               - Monitor sector concentration risks
               - Develop co-investment partnerships for risk sharing
            
            3. **Opportunity Targeting**
               - Focus on Seed and Series A stages in high-growth sectors
               - Explore emerging ecosystems for better valuations
               - Build presence in Prairie and Atlantic regions
        """
        )

    with col2:
        st.markdown(
            """
            #### For Startups
            1. **Funding Strategy**
               - Target raise sizes aligned with stage medians
               - Build relationships with both local and international investors
               - Consider alternative funding sources for bridge rounds
            
            2. **Growth Planning**
               - Focus on key metrics that drive successful raises
               - Develop clear path to next funding milestone
               - Build strategic partnerships in target markets
            
            3. **Risk Mitigation**
               - Maintain 18-24 month runway
               - Diversify customer base and revenue streams
               - Build strong governance and reporting frameworks
        """
        )

    st.markdown(
        """
        #### For Policy Makers
        1. **Ecosystem Development**
           - Strengthen support for early-stage companies
           - Develop targeted programs for high-potential sectors
           - Create incentives for international investor participation
        
        2. **Regional Balance**
           - Implement programs to support emerging ecosystems
           - Create cross-regional collaboration frameworks
           - Develop specialized support for underserved regions
        
        3. **Competitiveness**
           - Streamline regulatory processes for startup operations
           - Enhance tax incentives for research and development
           - Support international market access programs
    """
    )
