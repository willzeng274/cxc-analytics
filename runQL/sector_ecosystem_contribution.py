import pandas as pd
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def display_sector_ecosystem_contribution_heatmap(analysis):
    """Display heatmap of ecosystem contributions to each sector."""
    st.header("Ecosystem Contribution by Sector")
    st.markdown(
        """
        <div class='insight-box'>
        Heatmap visualization showing how much each ecosystem contributes to different sectors.
        This reveals which ecosystems are the major contributors to specific sectors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    companies_df = pd.DataFrame(analysis['raw_data']['companies'])
    
    top_sectors = companies_df['primaryTag'].value_counts().head(10).index.tolist()
    top_ecosystems = companies_df['ecosystemName'].value_counts().head(10).index.tolist()
    
    filtered_df = companies_df[
        (companies_df['primaryTag'].isin(top_sectors)) & 
        (companies_df['ecosystemName'].isin(top_ecosystems))
    ]
    
    sector_ecosystem_counts = pd.crosstab(
        filtered_df['ecosystemName'], 
        filtered_df['primaryTag']
    )
    
    # Calculate percentages by column (sector) instead of by row (ecosystem)
    sector_ecosystem_pct = sector_ecosystem_counts.div(sector_ecosystem_counts.sum(axis=0)) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=sector_ecosystem_pct.values,
        x=sector_ecosystem_pct.columns,
        y=sector_ecosystem_pct.index,
        colorscale='Viridis',
        text=sector_ecosystem_pct.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 12},
        hoverongaps=False,
        colorbar=dict(
            title=dict(
                text="Percentage",
                font=dict(size=14)
            ),
            ticksuffix="%"
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='ECOSYSTEM CONTRIBUTION TO SECTORS',
            font=dict(size=20, color='rgb(2, 33, 105)'),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title=dict(
                text="Sector",
                font=dict(size=16, color='rgb(50, 50, 50)')
            ),
            tickangle=-45,
            tickfont=dict(size=14, color='rgb(50, 50, 50)')
        ),
        yaxis=dict(
            title=dict(
                text="Ecosystem",
                font=dict(size=16, color='rgb(50, 50, 50)')
            ),
            tickfont=dict(size=14, color='rgb(50, 50, 50)')
        ),
        height=700,
        template='plotly_white',
        margin=dict(t=100, b=150, l=150, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔍 Key Insights")
    
    # Find top contributors for each sector
    sector_leaders = {}
    for sector in sector_ecosystem_pct.columns:
        top_ecosystem = sector_ecosystem_pct[sector].idxmax()
        top_pct = sector_ecosystem_pct[sector][top_ecosystem]
        sector_leaders[sector] = (top_ecosystem, top_pct)
    
    # Find sectors where ecosystems are dominant
    ecosystem_strengths = {}
    for ecosystem in sector_ecosystem_pct.index:
        top_sector = sector_ecosystem_pct.loc[ecosystem].idxmax()
        top_pct = sector_ecosystem_pct.loc[ecosystem, top_sector]
        ecosystem_strengths[ecosystem] = (top_sector, top_pct)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Leading Contributors by Sector")
        for sector, (ecosystem, pct) in sorted(sector_leaders.items(), key=lambda x: x[1][1], reverse=True)[:5]:
            st.markdown(f"- **{sector}**: {ecosystem} ({pct:.1f}%)")
    
    with col2:
        st.markdown("#### Ecosystem Leadership Areas")
        for ecosystem, (sector, pct) in sorted(ecosystem_strengths.items(), key=lambda x: x[1][1], reverse=True)[:5]:
            st.markdown(f"- **{ecosystem}**: {sector} ({pct:.1f}%)")
    
    st.markdown("#### Market Concentration Analysis")
    
    # Calculate concentration metrics for each sector
    concentration_metrics = {}
    for sector in sector_ecosystem_pct.columns:
        contributions = sector_ecosystem_pct[sector].sort_values(ascending=False)
        top_3_share = contributions[:3].sum()
        concentration_metrics[sector] = top_3_share
    
    st.markdown("**Top 3 Most Concentrated Sectors:**")
    for sector, concentration in sorted(concentration_metrics.items(), reverse=True)[:3]:
        st.markdown(f"- **{sector}**: Top 3 ecosystems account for {concentration:.1f}% of companies") 