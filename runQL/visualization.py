import io
import json
import traceback
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from plotly.subplots import make_subplots


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


def create_download_buttons(fig, filename_prefix, width=None, height=None):
    """Create download buttons for a plotly figure"""
    img_buffer = io.BytesIO()
    
    download_fig = fig.to_dict()
    download_fig = go.Figure(download_fig)
    
    if width or height:
        download_fig.update_layout(
            width=width if width else fig.layout.width * 5,
            height=height if height else fig.layout.height * 5,
            margin=dict(t=120, b=100)
        )
    
    if 'annotations' in fig.layout:
        download_fig.update_layout(annotations=fig.layout.annotations)
    
    download_fig.write_image(img_buffer, format="png")
    img_buffer.seek(0)

    st.download_button(
        label="Download Graph as PNG",
        data=img_buffer,
        file_name=f"{filename_prefix}.png",
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


def display_venture_capital_heatmap():
    """Display the venture capital investment data across Canada"""
    st.header("🍁 Canadian Venture Capital Investment Distribution")
    
    st.markdown(
        """
        <div class='insight-box'>
        Geographic distribution of venture capital investments across Canadian ecosystems.
        </div>
        """,
        unsafe_allow_html=True,
    )

    companies_df = analysis['raw_data']['companies']
    deals_df = analysis['raw_data']['deals']
    
    merged_df = deals_df.merge(
        companies_df[['id', 'ecosystemName', 'primaryTag']], 
        left_on='companyId', 
        right_on='id', 
        suffixes=('_deal', '')
    )
    
    st.subheader("VC Year-over-Year Sector Investment Activity (2020-2024)")
    
    merged_df['year'] = pd.to_datetime(merged_df['date']).dt.year
    
    sector_stats = merged_df.groupby(['primaryTag', 'year']).agg({
        'amount': ['sum', 'count']
    }).reset_index()
    
    sector_growth = {}
    for sector in sector_stats['primaryTag'].unique():
        sector_data = sector_stats[sector_stats['primaryTag'] == sector]
        total_investment = sector_data[('amount', 'sum')].sum()
        
        growth = 0.0
        
        years = sorted(sector_data['year'].unique())
        if len(years) >= 2:
            latest_year = max(years)
            latest_data = sector_data[sector_data['year'] == latest_year]
            prev_data = sector_data[sector_data['year'] == latest_year - 1]
            
            if not latest_data.empty and not prev_data.empty:
                latest_amount = latest_data[('amount', 'sum')].values[0]
                prev_amount = prev_data[('amount', 'sum')].values[0]
                
                if prev_amount > 0:
                    growth = ((latest_amount - prev_amount) / prev_amount) * 100
        
        sector_growth[sector] = {
            'growth': growth,
            'total_investment': total_investment
        }
    
    top_sectors = sorted(sector_growth.items(), key=lambda x: x[1]['total_investment'], reverse=True)[:5]
    top_sector_names = [sector[0] for sector in top_sectors]
    
    fig_sectors = go.Figure()
    
    bar_color = 'rgb(2, 33, 105)'
    line_colors = ['rgb(46, 204, 113)', 'rgb(231, 76, 60)', 'rgb(52, 152, 219)', 
                  'rgb(155, 89, 182)', 'rgb(241, 196, 15)']
    
    years = list(range(2020, 2025))
    x_positions = []
    x_ticks = []
    x_labels = []
    
    for i, sector in enumerate(top_sector_names):
        sector_x = []
        for year_idx, year in enumerate(years):
            pos = i * 6 + year_idx
            x_positions.append(pos)
            sector_x.append(pos)
            x_ticks.append(pos)
            x_labels.append(year)
    
    year_data_list = []
    for year_idx, year in enumerate(years):
        year_data = []
        year_x = []
        for i, sector in enumerate(top_sector_names):
            sector_data = sector_stats[sector_stats['primaryTag'] == sector]
            year_sector_data = sector_data[sector_data['year'] == year]
            investment_amount = year_sector_data[('amount', 'sum')].values[0] / 1e6 if not year_sector_data.empty else 0
            year_data.append(investment_amount)
            year_x.append(i * 6 + year_idx)
        year_data_list.append(year_data)
        
        fig_sectors.add_trace(go.Bar(
            name=str(year),
            x=year_x,
            y=year_data,
            marker_color=bar_color,
            opacity=0.6 + 0.1 * (year - 2020),
            text=[f'${round(x, 1)}M' for x in year_data],
            textposition='outside',
            textfont=dict(size=10, color='black'),
            textangle=270,
            cliponaxis=False,
            constraintext='none'
        ))
    
    all_deal_counts = []
    for i, sector in enumerate(top_sector_names):
        sector_data = sector_stats[sector_stats['primaryTag'] == sector]
        sector_data = sector_data.sort_values('year')
        
        sector_x = []
        deal_counts = []
        
        for year_idx, year in enumerate(years):
            year_data = sector_data[sector_data['year'] == year]
            count = year_data[('amount', 'count')].values[0] if not year_data.empty else 0
            deal_counts.append(count)
            sector_x.append(i * 6 + year_idx)
        
        all_deal_counts.extend(deal_counts)
        
        fig_sectors.add_trace(go.Scatter(
            name=f'{sector} Deals',
            x=sector_x,
            y=deal_counts,
            yaxis='y2',
            line=dict(color=line_colors[i], width=3),
            mode='lines+markers',
            marker=dict(size=8),
        ))
    
    max_investment = max(max(year_data) for year_data in year_data_list)
    max_deals = max(all_deal_counts)

    fig_sectors.update_layout(
        title=dict(
            text='TOP 5 SECTORS: INVESTMENT & DEAL VOLUME TRENDS<br>(2020-2024)',
            font=dict(size=24, color='rgb(2, 33, 105)'),
            x=0.5,
            y=0.97
        ),
        barmode='group',
        height=800,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        yaxis=dict(
            title='Investment Amount ($ Millions)',
            gridcolor='rgba(128, 128, 128, 0.2)',
            side='left',
            range=[0, max_investment * 1.1]
        ),
        yaxis2=dict(
            title='Number of Deals',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, max_deals * 1.1]
        ),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            tickangle=-45,
            tickmode='array',
            tickvals=x_ticks,
            ticktext=x_labels,
            tickfont=dict(size=11),
            domain=[0, 1]
        ),
        margin=dict(t=170, b=80, l=80, r=80)
    )
    
    for i, sector in enumerate(top_sector_names):
        if sector in sector_growth:
            growth_rate = sector_growth[sector]['growth']
            fig_sectors.add_annotation(
                x=i * 6 + 2,
                y=-0.15,
                text=sector,
                showarrow=False,
                yshift=0,
                xanchor='center',
                yref='paper',
                font=dict(size=11, color='rgb(2, 33, 105)')
            )
            fig_sectors.add_annotation(
                x=i * 6 + 2,
                y=-0.22,
                text=f"Growth: {growth_rate:.1f}%",
                showarrow=False,
                yshift=0,
                xanchor='center',
                yref='paper',
                font=dict(size=10, color='rgb(2, 33, 105)')
            )
    
    st.plotly_chart(fig_sectors, use_container_width=True)
    create_download_buttons(fig_sectors, "sector_growth_analysis", width=1600, height=1000)

    ecosystem_stats = merged_df.groupby('ecosystemName').agg({
        'id_deal': 'count',
        'amount': ['sum', 'mean']
    }).round(3)
    
    ecosystem_stats.columns = ['Number of Deals', 'Total Investment', 'Average Deal Size']
    ecosystem_stats = ecosystem_stats.reset_index()
    
    ecosystem_stats['Total Investment (B)'] = ecosystem_stats['Total Investment'] / 1e9
    ecosystem_stats['Average Deal Size (M)'] = ecosystem_stats['Average Deal Size'] / 1e6
    
    total_investment = ecosystem_stats['Total Investment'].sum() / 1e9
    total_deals = ecosystem_stats['Number of Deals'].sum()

    col1_, col2_ = st.columns(2)
    with col1_:
        st.metric("Total Investment", f"${total_investment:.1f}B", "Across Canada")
    with col2_:
        st.metric("Total Deals", f"{total_deals:,}", "Investment Rounds")

    st.subheader("🔍 Key Insights")
    
    top_deals = ecosystem_stats.nlargest(3, 'Number of Deals')
    top_investment = ecosystem_stats.nlargest(3, 'Total Investment (B)')
    
    col1_, col2_ = st.columns(2)
    
    with col1_:
        st.markdown("**Top 3 Ecosystems by Deal Volume:**")
        for _, row in top_deals.iterrows():
            pct = (row['Number of Deals'] / total_deals) * 100
            st.markdown(f"- {row['ecosystemName']}: {row['Number of Deals']} deals ({pct:.1f}%)")
        
        st.markdown("""
        **Investment Concentration:**
        """)
        top3_deals_pct = (top_deals['Number of Deals'].sum() / total_deals) * 100
        st.markdown(f"- Top 3 ecosystems account for {top3_deals_pct:.1f}% of all deals")
    
    with col2_:
        st.markdown("**Top 3 Ecosystems by Investment:**")
        for _, row in top_investment.iterrows():
            pct = (row['Total Investment'] / ecosystem_stats['Total Investment'].sum()) * 100
            amount = row['Total Investment (B)']
            amount_str = f"${amount:.1f}B" if amount >= 1 else f"${int(amount*1000)}M"
            st.markdown(f"- {row['ecosystemName']}: {amount_str} ({pct:.1f}%)")
        
        st.markdown("""
        **Capital Distribution:**
        """)
        top3_inv_pct = (top_investment['Total Investment'].sum() / ecosystem_stats['Total Investment'].sum()) * 100
        st.markdown(f"- Top 3 ecosystems capture {top3_inv_pct:.1f}% of total investment")

    ecosystem_stats_sorted = ecosystem_stats.sort_values('Number of Deals', ascending=True)
    
    summary_df = ecosystem_stats_sorted.copy()
    summary_df['Deal Share (%)'] = (summary_df['Number of Deals'] / total_deals * 100).round(1)
    summary_df['Investment Share (%)'] = (summary_df['Total Investment'] / ecosystem_stats['Total Investment'].sum() * 100).round(1)
    summary_df.sort_values('Number of Deals', ascending=False, inplace=True)
    
    summary_table = pd.DataFrame({
        'Ecosystem': summary_df['ecosystemName'],
        'Number of Deals': summary_df['Number of Deals'],
        'Deal Share (%)': summary_df['Deal Share (%)'].apply(lambda x: f"{x:.1f}%"),
        'Total Investment': summary_df['Total Investment (B)'].apply(
            lambda x: f"${x:.1f}B" if x >= 1 else f"${int(x*1000)}M"
        ),
        'Investment Share (%)': summary_df['Investment Share (%)'].apply(lambda x: f"{x:.1f}%"),
        'Average Deal Size': summary_df['Average Deal Size (M)'].apply(lambda x: f"${x:.1f}M")
    })
    
    st.markdown("### 📊 Detailed Metrics for Ecosystems")
    st.dataframe(
        summary_table,
        hide_index=True,
        column_config={
            "Ecosystem": st.column_config.TextColumn("Ecosystem", width="medium"),
            "Number of Deals": st.column_config.NumberColumn("Number of Deals", format="%d"),
            "Deal Share (%)": st.column_config.TextColumn("% of Total Deals", width="small"),
            "Total Investment": st.column_config.TextColumn("Total Investment", width="medium"),
            "Investment Share (%)": st.column_config.TextColumn("% of Total Investment", width="small"),
            "Average Deal Size": st.column_config.TextColumn("Avg Deal Size", width="medium")
        }
    )
    
    st.markdown("### 📊 Visual Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Number of Deals',
        x=ecosystem_stats_sorted['ecosystemName'],
        y=ecosystem_stats_sorted['Number of Deals'],
        marker_color='rgba(52, 152, 219, 0.8)'
    ))
    
    fig.add_trace(go.Bar(
        name='Average Deal Size (M)',
        x=ecosystem_stats_sorted['ecosystemName'],
        y=ecosystem_stats_sorted['Average Deal Size (M)'],
        marker_color='rgba(46, 204, 113, 0.8)'
    ))
    
    fig.update_layout(
        title='Deal Volume vs Average Deal Size by Ecosystem (Top 10)',
        barmode='group',
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    create_download_buttons(fig, "ecosystem_comparison")

    st.markdown("### 📈 Quarterly Investment Activity")
    
    try:
        quarterly_periods = pd.to_datetime(merged_df['date']).dt.to_period('Q')
        quarterly_periods.name = 'quarter'
        quarterly_stats = merged_df.groupby(quarterly_periods).agg({
            'id_deal': 'count',
            'amount': 'sum'
        })
        
        quarterly_stats = quarterly_stats.reset_index()
        
        quarterly_stats = quarterly_stats[quarterly_stats['quarter'].astype(str).str[:4].astype(int) < 2025]
        
        quarterly_stats['quarter_str'] = quarterly_stats['quarter'].astype(str)
        quarterly_stats = quarterly_stats.sort_values('quarter')

        stage_mapping = {
            'Pre-Seed': 'Seed',
            'Seed': 'Seed',
            'Series A': 'Early Stage',
            'Series B': 'Early Stage',
            'Series C': 'Later Stage',
            'Series D': 'Later Stage',
            'Series E': 'Later Stage',
            'Series F': 'Later Stage',
            'Series G': 'Later Stage',
            'Growth': 'Growth',
            'Private Equity': 'Growth'
        }

        if 'roundType' in merged_df.columns:
            merged_df['stage_category'] = merged_df['roundType'].map(stage_mapping)
            
            stage_quarterly_stats = {}
            for stage in ['Seed', 'Early Stage', 'Later Stage', 'Growth']:
                stage_data = merged_df[merged_df['stage_category'] == stage]
                if not stage_data.empty:
                    stage_stats = stage_data.groupby(quarterly_periods).agg({
                        'id_deal': 'count'
                    }).reset_index()
                    stage_stats = stage_stats[stage_stats['quarter'].astype(str).str[:4].astype(int) < 2025]  # Filter 2025
                    stage_stats['quarter_str'] = stage_stats['quarter'].astype(str)
                    stage_quarterly_stats[stage] = stage_stats
        else:
            st.warning("Round type information not available. Stage-specific trends will not be shown.")
            stage_quarterly_stats = {}
        
        fig_quarterly = go.Figure()

        fig_quarterly.add_trace(go.Bar(
            x=quarterly_stats['quarter_str'],
            y=quarterly_stats['amount'] / 1e6,
            name='Total Investment',
            marker_color='rgb(2, 33, 105)',
            width=0.7,
            text=[f'${round(x, 1)}M' for x in quarterly_stats['amount'] / 1e6],
            textposition='auto',
            textfont=dict(color='white', size=13),
            textangle=270,
            insidetextanchor='start',
            hovertemplate='<b>%{x}</b><br>Investment: $%{y:.1f}M<br><extra></extra>'
        ))

        fig_quarterly.add_trace(go.Scatter(
            x=quarterly_stats['quarter_str'],
            y=quarterly_stats['id_deal'],
            name='Total Deals',
            yaxis='y2',
            mode='lines+markers+text',
            line=dict(color='rgb(52, 152, 219)', width=2),
            marker=dict(size=8),
            text=quarterly_stats['id_deal'],
            textposition='top center',
            textfont=dict(size=12, color='rgba(52, 152, 219, 0.8)')
        ))

        colors = {
            'Seed': 'rgb(46, 204, 113)',
            'Early Stage': 'rgb(241, 196, 15)',
            'Later Stage': 'rgb(155, 89, 182)',
            'Growth': 'rgb(230, 126, 34)'
        }

        for stage, stats in stage_quarterly_stats.items():
            fig_quarterly.add_trace(go.Scatter(
                x=stats['quarter_str'],
                y=stats['id_deal'],
                name=f'{stage} Deals',
                mode='lines',
                yaxis='y2',
                line=dict(color=colors[stage], width=2),
            ))

        yearly_stats = merged_df.groupby(pd.to_datetime(merged_df['date']).dt.year).agg({
            'id_deal': 'count',
            'amount': 'sum'
        }).reset_index()
        yearly_stats = yearly_stats[yearly_stats['date'] < 2025]
        
        yearly_annotations = []
        for year in yearly_stats['date'].unique():
            year_data = yearly_stats[yearly_stats['date'] == year]
            deals = year_data['id_deal'].iloc[0]
            amount = year_data['amount'].iloc[0] / 1e9
            
            year_quarters = [f"{year}Q{q}" for q in range(1, 5)]
            if all(q in quarterly_stats['quarter_str'].values for q in year_quarters):
                x_pos = year_quarters[1]
                
                yearly_annotations.extend([
                    dict(
                        x=x_pos,
                        y=-0.15,
                        text=f"{year}<br>{deals} DEALS",
                        showarrow=False,
                        font=dict(size=10, color='rgb(52, 152, 219)'),
                        xanchor='center',
                        yanchor='top',
                        yref='paper',
                        xref='x'
                    ),
                    dict(
                        x=x_pos,
                        y=-0.25,
                        text=f"${amount:.1f}B",
                        showarrow=False,
                        font=dict(size=10, color='rgb(2, 33, 105)'),
                        xanchor='center',
                        yanchor='top',
                        yref='paper',
                        xref='x'
                    )
                ])

        fig_quarterly.update_layout(
            title=dict(
                text='VENTURE CAPITAL<br>INVESTMENT ACTIVITY',
                font=dict(size=24, color='rgb(2, 33, 105)'),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title="",
                tickangle=-45,
                showgrid=False,
                showline=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=2,
                type="category",
                tickmode='array',
                ticktext=[f"Q{q}" if q <= 4 else "" for year in range(2019, 2025) for q in range(1, 5)],
                tickvals=quarterly_stats['quarter_str']
            ),
            yaxis=dict(
                title="$ Millions Invested",
                showgrid=True,
                gridcolor="rgb(204, 204, 204)",
                tickformat="$,.0f",
                range=[0, max(quarterly_stats['amount'] / 1e6) * 1.2],
                showline=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=2
            ),
            yaxis2=dict(
                title="Number of Deals",
                overlaying="y",
                side="right",
                showgrid=False,
                range=[0, max(quarterly_stats['id_deal']) * 1.2],
                showline=True,
                linecolor="rgb(204, 204, 204)",
                linewidth=2
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            height=600,
            margin=dict(t=120, b=100),
            hovermode='x unified',
            annotations=yearly_annotations
        )

        fig_quarterly.update_yaxes(
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            griddash='dot'
        )

        st.plotly_chart(fig_quarterly, use_container_width=True)
        create_download_buttons(fig_quarterly, "quarterly_activity", width=1600, height=900)
    except Exception as e:
        st.error(f"Error in quarterly investment visualization: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")

    st.markdown("### 🏆 Top Investors by Deal Volume and Investment")
    
    deals_df = pd.DataFrame(analysis['raw_data']['deals'])
    investors_df = pd.DataFrame(analysis["raw_data"]["deal_investor"])
    
    merged_df = pd.merge(investors_df[['dealId', 'investorName', 'investorId']], deals_df[['id', 'amount']],
                        left_on='dealId', right_on='id', 
                        how='left')
    
    deal_counts = merged_df.groupby('dealId').size().reset_index(name='num_investors')
    merged_df = pd.merge(merged_df, deal_counts, on='dealId')
    
    merged_df['estimated_amount'] = merged_df['amount'] / merged_df['num_investors']
    
    investor_totals = merged_df.groupby(['investorName', 'investorId']).agg({
        'dealId': 'count',
        'estimated_amount': 'sum'
    }).reset_index()
    
    investor_totals.columns = ['Investor Name', 'Investor ID', 'Number of Deals', 'Total Investment']
    investor_totals['Total Investment (Millions)'] = (investor_totals['Total Investment'] / 1_000_000).round(1)
    
    top_investors = investor_totals.sort_values('Total Investment (Millions)', ascending=False).head(20)
    
    fig_top_investors = go.Figure()
    
    fig_top_investors.add_trace(go.Bar(
        y=top_investors['Investor Name'],
        x=top_investors["Total Investment (Millions)"],
        orientation='h',
        width=0.6,
        marker_color='rgb(52, 152, 219)',
        text=[f"${x:,.0f}M" for x in top_investors["Total Investment (Millions)"]],
        textposition='outside',
        textfont=dict(size=12)
    ))
    
    fig_top_investors.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=-0.5,
        y1=len(top_investors)-0.5,
        line=dict(color="rgb(128, 128, 128)", width=1)
    )
    
    for i, (_, row) in enumerate(top_investors.iterrows()):
        fig_top_investors.add_shape(
            type="line",
            x0=0,
            x1=0,
            y0=i-0.25,
            y1=i+0.25,
            line=dict(color="rgb(128, 128, 128)", width=1)
        )
        fig_top_investors.add_annotation(
            x=0,
            y=i,
            text=str(int(row['Number of Deals'])),
            showarrow=False,
            xanchor='right',
            xshift=-10,
            font=dict(size=12)
        )
    
    fig_top_investors.update_layout(
        title=dict(
            text='Size of Total Rounds* ($M)',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False,
            title='',
            tickformat='$,.0f',
            side='top'  # Move x-axis to top
        ),
        yaxis=dict(
            title=dict(
                text='# Rounds',
                standoff=25,
            ),
            showgrid=False,
            zeroline=False,
            autorange="reversed",
        ),
        height=600,
        margin=dict(l=250, r=150, t=50, b=20),
        plot_bgcolor='white',
        showlegend=False,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_top_investors, use_container_width=True)
    create_download_buttons(fig_top_investors, "top_investors_analysis", width=1600, height=600)


def display_funding_stage_breakdown():
    """Display funding stage breakdown visualization with three separate graphs."""
    st.header("📊 Funding Stage Breakdown")
    st.markdown(
        """
        <div class='insight-box'>
        Analysis of investment trends across different funding stages: Seed, Early-stage, and Late-stage.
        Note: Analysis includes data up to 2024 to ensure completeness and accuracy of trends.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get the deals data
    deals_df = pd.DataFrame(analysis['raw_data']['deals'])
    
    # Convert date to datetime and extract year and quarter
    deals_df['date'] = pd.to_datetime(deals_df['date'])
    deals_df['year'] = deals_df['date'].dt.year
    deals_df['quarter'] = deals_df['date'].dt.quarter
    deals_df['year_quarter'] = deals_df['date'].dt.to_period('Q')
    
    # Create funding series breakdown visualization

    st.subheader("💰 Investment by Funding Series")
    
    # Define funding series order for proper sorting
    series_order = ['Pre-Seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Series D', 'Series E', 'Series F', 'Series G']
    
    # Filter out data beyond 2024
    deals_df_filtered = deals_df[deals_df['year'] < 2025]
    
    # Create quarterly series aggregation
    series_quarterly = pd.pivot_table(
        deals_df_filtered,
        values='amount',
        index='year_quarter',
        columns='roundType',
        aggfunc='sum',
        fill_value=0
    ) / 1e6  # Convert to millions
    
    # Reorder columns based on series_order
    series_quarterly = series_quarterly[series_quarterly.columns.intersection(series_order)]
    
    # Create the stacked bar chart for funding series
    fig_series = go.Figure()
    
    # Define colors for funding series
    colors = {
        'Pre-Seed': 'rgb(198, 219, 239)',
        'Seed': 'rgb(158, 202, 225)',
        'Series A': 'rgb(107, 174, 214)',
        'Series B': 'rgb(66, 146, 198)',
        'Series C': 'rgb(33, 113, 181)',
        'Series D': 'rgb(8, 81, 156)',
        'Series E': 'rgb(8, 69, 148)',
        'Series F': 'rgb(8, 48, 107)',
        'Series G': '#111c50'
    }
    
    for column in series_quarterly.columns:
        fig_series.add_trace(go.Bar(
            name=column,
            x=series_quarterly.index.astype(str),
            y=series_quarterly[column],
            marker_color=colors.get(column, 'rgb(128, 128, 128)'),
            # text=[f'${x:.1f}M' if x > 0 else '' for x in series_quarterly[column]],
            # textposition='inside',
            # insidetextanchor='start',
            # textangle=270,
            # textfont=dict(size=10),
            # hovertemplate='%{x}<br>' + column + ': $%{y:.1f}M<extra></extra>'
        ))
    
    # Add total investment annotations at the top of each stacked bar
    quarterly_totals = series_quarterly.sum(axis=1)
    for x, total in zip(series_quarterly.index.astype(str), quarterly_totals):
        fig_series.add_annotation(
            x=x,
            y=total,
            text=f'${total:.1f}M',
            showarrow=False,
            yshift=10,
            font=dict(size=11.5, color='#111c50')
        )
    
    fig_series.update_layout(
        title=dict(
            text='QUARTERLY INVESTMENT BY FUNDING SERIES',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        barmode='stack',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title='',
            tickangle=-45,
            tickmode='array',
            ticktext=[f"Q{q}" if q <= 4 else "" for year in range(2019, 2025) for q in range(1, 5)],
            tickvals=series_quarterly.index.astype(str)
        ),
        yaxis=dict(
            title='Investment Amount ($ Millions)',
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        height=600,
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig_series, use_container_width=True)
    create_download_buttons(fig_series, "_breakdown")

    st.subheader("📈 Investment by Deal Size")
    
    # Define funding series categories based on actual amounts
    deals_df['series_category'] = pd.cut(
        deals_df['amount'] / 1e6,  # Convert to millions
        bins=[-float('inf'), 5, 15, 40, 100, 250, float('inf')],
        labels=['$0-5M', '$5-15M', '$15-40M', '$40-100M', '$100-250M', '$250M+']
    )
    
    # Filter out data beyond 2024
    deals_df_filtered = deals_df[deals_df['year'] < 2025]
    
    # Create quarterly series aggregation
    series_quarterly = pd.pivot_table(
        deals_df_filtered,
        values='amount',
        index='year_quarter',
        columns='series_category',
        aggfunc='sum',
        fill_value=0
    ) / 1e6  # Convert to millions
    
    # Create the stacked bar chart for funding series
    fig_series = go.Figure()
    
    # Define colors for funding series
    colors = {
        '$0-5M': 'rgb(158, 202, 225)',
        '$5-15M': 'rgb(107, 174, 214)',
        '$15-40M': 'rgb(66, 146, 198)',
        '$40-100M': 'rgb(33, 113, 181)',
        '$100-250M': 'rgb(8, 81, 156)',
        '$250M+': '#111c50'
    }
    
    for column in series_quarterly.columns:
        fig_series.add_trace(go.Bar(
            name=column,
            x=series_quarterly.index.astype(str),
            y=series_quarterly[column],
            marker_color=colors[column],
            # text=[f'${x:.1f}M' if x > 0 else '' for x in series_quarterly[column]],
            # textposition='inside',
            # insidetextanchor='start',
            # textangle=270,
            # textfont=dict(size=10),
            # hovertemplate='%{x}<br>' + column + ': $%{y:.1f}M<extra></extra>'
        ))
    
    # Add total investment annotations at the top of each stacked bar
    quarterly_totals = series_quarterly.sum(axis=1)
    for x, total in zip(series_quarterly.index.astype(str), quarterly_totals):
        fig_series.add_annotation(
            x=x,
            y=total,
            text=f'${total:.1f}M',
            showarrow=False,
            yshift=10,
            font=dict(size=11.5, color='#111c50')
        )
    
    fig_series.update_layout(
        title=dict(
            text='QUARTERLY INVESTMENT BY DEAL SIZE',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        barmode='stack',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title='',
            tickangle=-45,
            tickmode='array',
            ticktext=[f"Q{q}" if q <= 4 else "" for year in range(2019, 2025) for q in range(1, 5)],
            tickvals=series_quarterly.index.astype(str)
        ),
        yaxis=dict(
            title='Investment Amount ($ Millions)',
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        height=600,
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig_series, use_container_width=True)
    create_download_buttons(fig_series, "funding_series_breakdown")
    
    st.subheader("📊 Median Deal Size & Volume Trends by Stage")
    
    # Calculate median deal size and count by year and series
    yearly_series_stats = deals_df_filtered.groupby(['year', 'roundType']).agg({
        'amount': ['median', 'count']
    }).reset_index()
    
    yearly_series_stats.columns = ['year', 'roundType', 'median_amount', 'deal_count']
    yearly_series_stats['median_amount'] = yearly_series_stats['median_amount'] / 1e6  # Convert to millions
    
    # Define stage mapping
    stage_mapping = {
        'Pre-Seed': 'Seeds',
        'Seed': 'Seeds',
        'Series A': 'Early',
        'Series B': 'Early',
        'Series C': 'Late',
        'Series D': 'Late',
        'Series E': 'Late',
        'Series F': 'Late',
        'Series G': 'Late'
    }
    
    # Add stage category and group by stage
    yearly_series_stats['stage'] = yearly_series_stats['roundType'].map(stage_mapping)
    stage_stats = yearly_series_stats.groupby(['year', 'stage']).agg({
        'median_amount': 'median',  # median of medians for the stage
        'deal_count': 'sum'  # sum of deals for the stage
    }).reset_index()
    
    # Color scheme for the stages
    stage_colors = {
        'Seeds': 'rgb(158, 202, 225)',
        'Early': 'rgb(66, 146, 198)',
        'Late': 'rgb(8, 81, 156)'
    }
    
    # Create median deal size visualization
    fig_medians = go.Figure()
    
    for stage in ['Seeds', 'Early', 'Late']:
        stage_data = stage_stats[stage_stats['stage'] == stage]
        fig_medians.add_trace(go.Scatter(
            x=stage_data['year'],
            y=stage_data['median_amount'],
            name=stage,
            mode='lines+markers',
            line=dict(color=stage_colors[stage], width=3),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Median: $%{y:.1f}M<extra></extra>'
        ))
    
    fig_medians.update_layout(
        title=dict(
            text='MEDIAN DEAL SIZE BY STAGE',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title='Year',
            tickmode='array',
            ticktext=[str(year) for year in range(2019, 2025)],
            tickvals=list(range(2019, 2025)),
            showgrid=False
        ),
        yaxis=dict(
            title='Median Deal Size ($ Millions)',
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickformat='$,.0f'
        ),
        height=600,
        margin=dict(t=100, b=50, l=80, r=50),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_medians, use_container_width=True)
    create_download_buttons(fig_medians, "median_deal_size_by_stage")

    # Create deal count visualization
    fig_counts = go.Figure()
    
    for stage in ['Seeds', 'Early', 'Late']:
        stage_data = stage_stats[stage_stats['stage'] == stage]
        fig_counts.add_trace(go.Scatter(
            x=stage_data['year'],
            y=stage_data['deal_count'],
            name=stage,
            mode='lines+markers',
            line=dict(color=stage_colors[stage], width=3),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Deals: %{y}<extra></extra>'
        ))
    
    fig_counts.update_layout(
        title=dict(
            text='NUMBER OF DEALS BY STAGE',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title='Year',
            tickmode='array',
            ticktext=[str(year) for year in range(2019, 2025)],
            tickvals=list(range(2019, 2025)),
            showgrid=False
        ),
        yaxis=dict(
            title='Number of Deals',
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        height=400,
        margin=dict(t=100, b=50, l=80, r=50),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_counts, use_container_width=True)
    create_download_buttons(fig_counts, "deal_count_by_stage")

    # Add the original three separate graphs for stage breakdown
    # Define stage mapping
    stage_mapping = {
        'Pre-Seed': 'Seed',
        'Seed': 'Seed',
        'Series A': 'Early Stage',
        'Series B': 'Early Stage',
        'Series C': 'Later Stage',
        'Series D': 'Later Stage',
        'Series E': 'Later Stage',
        'Series F': 'Later Stage',
        'Series G': 'Later Stage',
    }
    
    # Map stages
    deals_df['stage_category'] = deals_df['roundType'].map(stage_mapping)
    
    # Create quarterly aggregations for each stage category
    def create_stage_data(df, stage_name):
        stage_data = df[df['stage_category'] == stage_name].groupby('year_quarter').agg({
            'amount': ['sum', 'count']
        }).reset_index()
        stage_data.columns = ['quarter', 'total_amount', 'deal_count']
        stage_data = stage_data[stage_data['quarter'].astype(str).str[:4].astype(int) < 2025]
        return stage_data

    seed_data = create_stage_data(deals_df, "Seed")
    early_data = create_stage_data(deals_df, "Early Stage")
    late_data = create_stage_data(deals_df, "Later Stage")

    # Create visualizations for individual stages
    def create_stage_chart(data, title, color_bar, color_line):
        fig = go.Figure()
        
        # Add bar chart for total amount
        fig.add_trace(go.Bar(
            x=data['quarter'].astype(str),
            y=data['total_amount'] / 1e6,  # Convert to millions
            name='Total Investment',
            marker_color=color_bar,
            opacity=0.8,
            text=[f'${x:.1f}M' for x in data['total_amount'] / 1e6],
            textposition='inside',
            textangle=270,
            insidetextanchor='start',
            textfont=dict(color='black', size=12),
            constraintext='none'
        ))
        
        # Add line chart for deal count
        fig.add_trace(go.Scatter(
            x=data['quarter'].astype(str),
            y=data['deal_count'],
            name='Number of Deals',
            yaxis='y2',
            line=dict(color=color_line, width=3),
            mode='lines+markers',
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{title}',
                font=dict(size=16),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(
                title='Investment Amount (Millions $)',
                gridcolor='rgba(128, 128, 128, 0.2)',
                side='left',
            ),
            yaxis2=dict(
                title='Number of Deals',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            height=600,
            margin=dict(t=100, b=50, l=50, r=50),
            hovermode='x unified'
        )
        
        # Update x-axis to show only Q1-Q4 labels
        fig.update_xaxes(
            ticktext=[f"Q{q}" if q <= 4 else "" for year in range(2019, 2025) for q in range(1, 5)],
            tickvals=data['quarter'].astype(str),
            tickangle=-45
        )
        
        return fig

    st.subheader("📊 Individual Stage Analysis")
    
    # Create and display the three charts
    seed_fig = create_stage_chart(seed_data, "Seeds-stage", "rgba(153, 204, 255, 0.8)", "rgba(0, 102, 204, 1)")
    st.plotly_chart(seed_fig, use_container_width=True)
    create_download_buttons(seed_fig, "seed_stage_analysis")

    early_fig = create_stage_chart(early_data, "Early-stage", "rgba(153, 255, 153, 0.8)", "rgba(0, 153, 0, 1)")
    st.plotly_chart(early_fig, use_container_width=True)
    create_download_buttons(early_fig, "early_stage_analysis")

    late_fig = create_stage_chart(late_data, "Late-stage", "rgba(255, 153, 153, 0.8)", "rgba(204, 0, 0, 1)")
    st.plotly_chart(late_fig, use_container_width=True)
    create_download_buttons(late_fig, "late_stage_analysis")


def display_company_age_analysis():
    """Display analysis of company age vs funding rounds and deal sizes."""
    st.header("🎯 Company Age & Funding Analysis")
    st.markdown(
        """
        <div class='insight-box'>
        Analysis of company age at different funding stages and its relationship with deal sizes.
        This can help founders understand typical timing for different funding rounds and sector growth patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get the necessary data
    companies_df = pd.DataFrame(analysis['raw_data']['companies'])
    deals_df = pd.DataFrame(analysis['raw_data']['deals'])
    
    # Merge deals with companies
    merged_df = deals_df.merge(
        companies_df[['id', 'dateFounded', 'primaryTag']], 
        left_on='companyId', 
        right_on='id', 
        suffixes=('_deal', '')
    )
    
    # Convert dates and calculate age
    merged_df['dateFounded'] = pd.to_datetime(merged_df['dateFounded'])
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['company_age'] = (merged_df['date'] - merged_df['dateFounded']).dt.days / 365.25  # Age in years
    
    # Create age categories
    age_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
    age_labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10+']
    merged_df['age_category'] = pd.cut(merged_df['company_age'], bins=age_bins, labels=age_labels)
    
    # Define stage categories
    stage_mapping = {
        'Pre-Seed': 'Seed',
        'Seed': 'Seed',
        'Series A': 'Early Stage',
        'Series B': 'Early Stage',
        'Series C': 'Late Stage',
        'Series D': 'Late Stage',
        'Series E': 'Late Stage',
        'Series F': 'Late Stage',
        'Series G': 'Late Stage'
    }
    merged_df['stage_category'] = merged_df['roundType'].map(stage_mapping)
    
    # Create age distribution visualization
    st.subheader("📊 Company Age Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by count
        age_dist = merged_df.groupby('age_category').size().reset_index(name='count')
        
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            x=age_dist['age_category'],
            y=age_dist['count'],
            marker_color='rgba(52, 152, 219, 0.8)',
            text=age_dist['count'],
            textposition='auto',
        ))
        
        fig_age.update_layout(
            title="Number of Companies by Age",
            xaxis_title="Company Age (Years)",
            yaxis_title="Number of Companies",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_age, use_container_width=True)
        create_download_buttons(fig_age, "age_distribution")
    
    with col2:
        # Age distribution by funding stage
        stage_age = pd.crosstab(merged_df['age_category'], merged_df['stage_category'])
        
        fig_stage = go.Figure()
        colors = {'Seed': 'rgba(46, 204, 113, 0.8)', 
                 'Early Stage': 'rgba(52, 152, 219, 0.8)', 
                 'Late Stage': 'rgba(155, 89, 182, 0.8)'}
        
        for stage in ['Seed', 'Early Stage', 'Late Stage']:
            if stage in stage_age.columns:
                fig_stage.add_trace(go.Bar(
                    name=stage,
                    x=stage_age.index,
                    y=stage_age[stage],
                    marker_color=colors[stage]
                ))
        
        fig_stage.update_layout(
            title="Funding Stages by Company Age",
            xaxis_title="Company Age (Years)",
            yaxis_title="Number of Deals",
            template='plotly_white',
            height=400,
            barmode='stack'
        )
        
        st.plotly_chart(fig_stage, use_container_width=True)
        create_download_buttons(fig_stage, "stage_by_age")
    
    # Sector growth analysis
    st.subheader("📈 Sector Growth by Company Age")
    
    # Calculate sector growth rates
    sector_age_stats = merged_df.groupby(['primaryTag', 'age_category']).agg({
        'amount': ['count', 'sum', 'mean']
    }).round(2)
    
    sector_age_stats.columns = ['deal_count', 'total_funding', 'avg_funding']
    sector_age_stats = sector_age_stats.reset_index()
    
    # Get top sectors by deal count
    top_sectors = sector_age_stats.groupby('primaryTag')['deal_count'].sum().nlargest(10).index
    
    # Create heatmap for top sectors
    pivot_data = sector_age_stats[sector_age_stats['primaryTag'].isin(top_sectors)].pivot(
        index='primaryTag',
        columns='age_category',
        values='deal_count'
    ).fillna(0)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='Viridis',
        text=pivot_data.values.astype(int),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
    ))
    
    fig_heatmap.update_layout(
        title="Deal Activity Heatmap: Sectors vs Company Age",
        xaxis_title="Company Age (Years)",
        yaxis_title="Sector",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    create_download_buttons(fig_heatmap, "sector_age_heatmap")
    
    # Summary statistics
    st.subheader("📊 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Age Distribution by Stage")
        stage_stats = merged_df.groupby('stage_category').agg({
            'company_age': ['median', 'mean', 'count']
        }).round(1)
        
        stage_stats.columns = ['Median Age', 'Mean Age', 'Number of Deals']
        stage_stats = stage_stats.reset_index()
        
        st.dataframe(
            stage_stats,
            hide_index=True,
            column_config={
                "stage_category": "Funding Stage",
                "Median Age": st.column_config.NumberColumn("Median Age (Years)", format="%.1f"),
                "Mean Age": st.column_config.NumberColumn("Mean Age (Years)", format="%.1f"),
                "Number of Deals": st.column_config.NumberColumn("Deal Count", format="%d")
            }
        )
    
    with col2:
        st.markdown("#### Top Growing Sectors by Age")
        sector_growth = sector_age_stats.groupby('primaryTag').agg({
            'deal_count': 'sum',
            'total_funding': 'sum',
            'avg_funding': 'mean'
        }).round(1)
        
        sector_growth = sector_growth.nlargest(10, 'deal_count').reset_index()
        
        st.dataframe(
            sector_growth,
            hide_index=True,
            column_config={
                "primaryTag": "Sector",
                "deal_count": st.column_config.NumberColumn("Total Deals", format="%d"),
                "total_funding": st.column_config.NumberColumn("Total Funding ($M)", format="$%.1fM"),
                "avg_funding": st.column_config.NumberColumn("Avg Deal Size ($M)", format="$%.1fM")
            }
        )


def display_stage_sector_analysis():
    """Display analysis of funding stages vs sectors over time."""
    st.header("📊 Funding Stages vs Sectors Analysis")
    st.markdown(
        """
        <div class='insight-box'>
        Analysis of sector maturity through funding stages, showing which sectors are maturing and where future growth opportunities may lie.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get the necessary data
    companies_df = pd.DataFrame(analysis['raw_data']['companies'])
    deals_df = pd.DataFrame(analysis['raw_data']['deals'])
    
    # Merge deals with companies
    merged_df = deals_df.merge(
        companies_df[['id', 'primaryTag']], 
        left_on='companyId', 
        right_on='id', 
        suffixes=('_deal', '')
    )
    
    # Convert dates and add year
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['year'] = merged_df['date'].dt.year
    
    # Define stage mapping
    stage_mapping = {
        'Pre-Seed': 'Seed',
        'Seed': 'Seed',
        'Series A': 'Early',
        'Series B': 'Early',
        'Series C': 'Late',
        'Series D': 'Late',
        'Series E': 'Late',
        'Series F': 'Late',
        'Series G': 'Late'
    }
    
    # Add stage category
    merged_df['stage_category'] = merged_df['roundType'].map(stage_mapping)
    
    # Get top 5 sectors by total deal count
    top_sectors = merged_df.groupby('primaryTag').size().nlargest(5).index
    
    # Filter for top sectors and years 2019-2024
    sector_data = merged_df[
        (merged_df['primaryTag'].isin(top_sectors)) & 
        (merged_df['year'].between(2019, 2024))
    ]
    
    # Create yearly funding series analysis
    st.subheader("📈 Yearly Investment by Funding Stage")
    
    # Define colors for stages
    colors = {
        'Seed': 'rgb(158, 202, 225)',
        'Early': 'rgb(66, 146, 198)',
        'Late': 'rgb(8, 81, 156)'
    }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['2019', '2020', '2021', '2022', '2023', '2024'],
        vertical_spacing=0.2,
        horizontal_spacing=0.05
    )
    
    # Calculate maximum y value for consistent scaling
    max_deals = 0
    for year in range(2019, 2025):
        year_data = sector_data[sector_data['year'] == year]
        stage_sector_counts = pd.crosstab(year_data['primaryTag'], year_data['stage_category'])
        max_deals = max(max_deals, stage_sector_counts.sum(axis=1).max())
    
    # Round max_deals to nearest 10 for clean y-axis
    max_deals = ((max_deals // 10) + 1) * 10
    
    # Create subplot for each year
    for i, year in enumerate(range(2019, 2025)):
        row = i // 3 + 1
        col = i % 3 + 1
        
        year_data = sector_data[sector_data['year'] == year]
        
        # Create crosstab for this year
        stage_sector_counts = pd.crosstab(
            year_data['primaryTag'],
            year_data['stage_category']
        ).fillna(0)
        
        # Sort sectors by total deals for this year
        stage_sector_counts['total'] = stage_sector_counts.sum(axis=1)
        stage_sector_counts = stage_sector_counts.sort_values('total', ascending=False)  # Changed to descending
        stage_sector_counts = stage_sector_counts.drop('total', axis=1)
        
        # Create stacked bars
        for j, stage in enumerate(['Seed', 'Early', 'Late']):
            if stage in stage_sector_counts.columns:
                fig.add_trace(
                    go.Bar(
                        name=stage,
                        x=stage_sector_counts.index,  # Swapped x and y
                        y=stage_sector_counts[stage],  # Swapped x and y
                        marker_color=colors[stage],
                        showlegend=(i == 0),  # Show legend only for first subplot
                        legendgroup=stage,
                    ),
                    row=row,
                    col=col
                )
    
        # Update axes for this subplot
        fig.update_xaxes(
            title=None,
            showgrid=False,
            zeroline=True,
            zerolinecolor='rgba(128, 128, 128, 0.2)',
            zerolinewidth=1,
            showline=True,
            linewidth=1,
            linecolor='rgba(128, 128, 128, 0.2)',
            tickangle=45,  # Added angle for better readability
            row=row,
            col=col
        )
        
        fig.update_yaxes(
            title=None,
            range=[0, max_deals],  # Moved range to y-axis
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinecolor='rgba(128, 128, 128, 0.2)',
            zerolinewidth=1,
            showline=True,
            linewidth=1,
            linecolor='rgba(128, 128, 128, 0.2)',
            row=row,
            col=col
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='FUNDING STAGE DISTRIBUTION BY SECTOR AND YEAR',
            font=dict(size=24, color='rgb(2, 33, 105)'),
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
            traceorder='normal'
        ),
        height=900,  # Increased from 800 to 900
        barmode='stack',
        template='plotly_white',
        plot_bgcolor='white',
        margin=dict(t=150, b=80, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    create_download_buttons(fig, "yearly_funding_series", width=1600, height=900)


def display_ecosystem_sector_trends():
    """Display sector funding trends for each ecosystem."""
    st.header("🌍 Ecosystem Sector Funding Trends")
    st.markdown(
        """
        <div class='insight-box'>
        Analysis of top sectors' funding distribution within each ecosystem over time, showing regional specialization and evolution.
        Note: Graph shows sectors with up to 60% of total funding to better highlight distribution patterns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get the necessary data
    companies_df = pd.DataFrame(analysis['raw_data']['companies'])
    deals_df = pd.DataFrame(analysis['raw_data']['deals'])
    
    # Merge deals with companies
    merged_df = deals_df.merge(
        companies_df[['id', 'primaryTag', 'ecosystemName']], 
        left_on='companyId', 
        right_on='id', 
        suffixes=('_deal', '')
    )
    
    # Convert dates and add year
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['year'] = merged_df['date'].dt.year
    
    # Filter years
    merged_df = merged_df[merged_df['year'].between(2019, 2024)]
    
    # Get top ecosystems by total funding
    top_ecosystems = merged_df.groupby('ecosystemName')['amount'].sum().nlargest(6).index
    
    # Create color scale for sectors
    colors = px.colors.qualitative.Set3
    
    # Process each ecosystem
    for ecosystem in top_ecosystems:
        st.subheader(f"📊 {ecosystem}")
        
        # Filter data for this ecosystem
        ecosystem_data = merged_df[merged_df['ecosystemName'] == ecosystem]
        
        # Process each year
        yearly_data = []
        for year in range(2019, 2025):
            year_data = ecosystem_data[ecosystem_data['year'] == year]
            
            # Calculate total funding for the year
            total_funding = year_data['amount'].sum()
            
            # Get top 5 sectors by funding for this year
            sector_funding = year_data.groupby('primaryTag')['amount'].sum()
            sector_percentages = (sector_funding / total_funding * 100).nlargest(5)
            
            # Store data
            for sector, percentage in sector_percentages.items():
                yearly_data.append({
                    'year': year,
                    'sector': sector,
                    'percentage': percentage
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(yearly_data)
        
        # Get unique sectors for color mapping
        unique_sectors = df['sector'].unique()
        sector_colors = {sector: colors[i % len(colors)] for i, sector in enumerate(unique_sectors)}
        
        # Create figure for this ecosystem
        fig = go.Figure()
        
        # Plot lines for each sector
        for sector in unique_sectors:
            sector_data = df[df['sector'] == sector]
            if not sector_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sector_data['year'],
                        y=sector_data['percentage'],
                        name=sector,
                        mode='lines+markers',
                        line=dict(color=sector_colors[sector], width=2),
                        marker=dict(size=6),
                        hovertemplate="%{y:.1f}%<extra>" + sector + "</extra>"
                    )
                )
        
        # Update layout for this ecosystem's figure
        fig.update_layout(
            title=dict(
                text=f'SECTOR FUNDING DISTRIBUTION - {ecosystem}',
                font=dict(size=16, color='rgb(2, 33, 105)'),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title='Year',
                tickmode='array',
                ticktext=['2019', '2020', '2021', '2022', '2023', '2024'],
                tickvals=[2019, 2020, 2021, 2022, 2023, 2024],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            yaxis=dict(
                title='% of Total Funding',
                range=[0, 100] if ecosystem == 'Waterloo Region' or ecosystem == 'Ottawa' else [0, 60],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                ticksuffix="%"
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            ),
            height=500,
            template='plotly_white',
            plot_bgcolor='white',
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        create_download_buttons(fig, f"ecosystem_sector_trends_{ecosystem.lower().replace(' ', '_')}")
        
        # Add a divider between ecosystems
        st.markdown("---")


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
        "Venture Capital Heat Map",
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
    
    total_investment = analysis['raw_data']['deals']['amount'].sum() / 1e9
    total_deals = len(analysis['raw_data']['deals'])
    total_companies = len(analysis['raw_data']['companies'])
    total_ecosystems = len(analysis['raw_data']['ecosystems'])
    year_range = f"{analysis['raw_data']['companies']['dateFounded'].dt.year.min()} - {analysis['raw_data']['companies']['dateFounded'].dt.year.max()}"
    
    metrics_df = pd.DataFrame({
        'Metric': [
            'Total Investment',
            'Total Deals',
            'Total Companies',
            'Total Ecosystems',
            'Time Period',
            'Data Sources'
        ],
        'Value': [
            f'${total_investment:.1f}B',
            f'{total_deals:,}',
            f'{total_companies:,}',
            str(total_ecosystems),
            year_range,
            'Multiple'
        ],
        'Description': [
            'Total capital invested',
            'Investment rounds',
            'Active startups',
            'Regional hubs',
            'Years covered',
            'Integrated sources'
        ]
    })
    
    st.dataframe(
        metrics_df,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="medium")
        }
    )

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

    # Add the new company age analysis
    display_company_age_analysis()
    
    # Add the funding stage breakdown visualization
    display_funding_stage_breakdown()

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
            top_investors = pd.DataFrame(analysis["investor_demographics"]["active_investors"]) 
            top_investors = top_investors.sort_values(by="dealId", ascending=False)
            top_investors = top_investors.head(20)
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
    
    # Add the new ecosystem sector trends visualization
    display_ecosystem_sector_trends()
    
    # Add sector funding heatmaps
    st.subheader("📊 Sector Funding Patterns")
    st.markdown(
        """
        Analysis of funding patterns across sectors over time, showing both deal volume and investment amounts.
        This helps identify growing sectors and changing investment preferences.
        """
    )
    
    # Get the necessary data
    companies_df = pd.DataFrame(analysis['raw_data']['companies'])
    deals_df = pd.DataFrame(analysis['raw_data']['deals'])
    
    # Merge deals with companies
    merged_df = deals_df.merge(
        companies_df[['id', 'primaryTag']], 
        left_on='companyId', 
        right_on='id', 
        suffixes=('_deal', '')
    )
    
    # Convert dates and add year
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['year'] = merged_df['date'].dt.year
    
    # Filter data before 2025
    merged_df = merged_df[merged_df['year'] < 2025]
    
    # Get top 10 sectors by total investment
    top_sectors = merged_df.groupby('primaryTag')['amount'].sum().nlargest(10).index
    
    # Create pivot tables for both metrics
    deals_pivot = pd.pivot_table(
        merged_df[merged_df['primaryTag'].isin(top_sectors)],
        values='id_deal',
        index='year',
        columns='primaryTag',
        aggfunc='count',
        fill_value=0
    )
    
    amount_pivot = pd.pivot_table(
        merged_df[merged_df['primaryTag'].isin(top_sectors)],
        values='amount',
        index='year',
        columns='primaryTag',
        aggfunc='sum',
        fill_value=0
    ) / 1e6  # Convert to millions
    
    # Create deal count heatmap
    fig_deals = go.Figure(data=go.Heatmap(
        z=deals_pivot.values,
        x=deals_pivot.columns,
        y=deals_pivot.index,
        colorscale='Viridis',
        text=deals_pivot.values.astype(int),
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False,
    ))
    
    fig_deals.update_layout(
        title=dict(
            text='NUMBER OF DEALS BY SECTOR AND YEAR',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Sector",
        yaxis_title="Year",
        height=500,
        template='plotly_white',
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig_deals, use_container_width=True)
    create_download_buttons(fig_deals, "sector_deals_heatmap")
    
    # Create investment amount heatmap
    fig_amount = go.Figure(data=go.Heatmap(
        z=amount_pivot.values,
        x=amount_pivot.columns,
        y=amount_pivot.index,
        colorscale='Viridis',
        text=amount_pivot.values.round(1),
        texttemplate="$%{text}M",
        textfont={"size": 12},
        hoverongaps=False,
    ))
    
    fig_amount.update_layout(
        title=dict(
            text='TOTAL INVESTMENT BY SECTOR AND YEAR ($ MILLIONS)',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Sector",
        yaxis_title="Year",
        height=500,
        template='plotly_white',
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig_amount, use_container_width=True)
    create_download_buttons(fig_amount, "sector_investment_heatmap")

    display_stage_sector_analysis()
    
    # Add the new stage-sector analysis
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

elif page == "Venture Capital Heat Map":
    display_venture_capital_heatmap()

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
