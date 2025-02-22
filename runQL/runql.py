import json
from datetime import datetime
import pandas as pd


def load_data():
    """Load and preprocess the datasets"""
    companies_df = pd.read_csv("runQL/data/companies.csv")
    ecosystems_df = pd.read_csv("runQL/data/ecosystems.csv")

    date_columns = [
        "dateFounded",
        "latestRoundDate",
        "dateAcqusition",
        "ipoDate",
        "peDate",
    ]
    for col in date_columns:
        companies_df[col] = pd.to_datetime(companies_df[col], errors="coerce")

    companies_df["age"] = (
        datetime.now() - companies_df["dateFounded"]
    ).dt.days / 365.25

    return companies_df, ecosystems_df


def get_ecosystem_stats(companies_df):
    """Generate ecosystem-level statistics"""
    ecosystem_stats = (
        companies_df.groupby("ecosystemName")
        .agg(
            {
                "id": "count",
                "age": "mean",
                "primaryTag": lambda x: x.value_counts().index[0],
                "latestRoundType": lambda x: x.value_counts().index[0],
            }
        )
        .reset_index()
    )

    ecosystem_stats.columns = [
        "ecosystemName",
        "company_count",
        "avg_company_age",
        "dominant_sector",
        "common_round_type",
    ]

    return ecosystem_stats


def get_sector_analysis(companies_df):
    """Analyze sector distribution and trends"""
    sector_stats = pd.DataFrame(
        {
            "primary_sectors": companies_df["primaryTag"].value_counts(),
            "secondary_sectors": companies_df["secondaryTag"].value_counts(),
        }
    ).fillna(0)

    companies_df["year_founded"] = companies_df["dateFounded"].dt.year
    sector_growth = (
        companies_df.groupby(["year_founded", "primaryTag"])
        .size()
        .unstack(fill_value=0)
    )

    return sector_stats, sector_growth


def get_funding_analysis(companies_df):
    """Analyze funding rounds and patterns"""
    round_distribution = companies_df["latestRoundType"].value_counts()

    funding_timeline = (
        companies_df.groupby("latestRoundDate")["id"].count().reset_index()
    )
    funding_timeline.columns = ["date", "number_of_rounds"]

    return round_distribution, funding_timeline


def analyze_investment_trends(deals_df):
    """Analyze investment trends over time"""
    deals_df["amount"] = pd.to_numeric(deals_df["amount"], errors="coerce")
    deals_df = deals_df[deals_df["year"] < 2025]

    deals_df["deal_size_category"] = pd.cut(
        deals_df["amount"],
        bins=[
            0,
            100000,
            500000,
            1000000,
            5000000,
            10000000,
            50000000,
            100000000,
            float("inf"),
        ],
        labels=[
            "<100K",
            "100K-500K",
            "500K-1M",
            "1M-5M",
            "5M-10M",
            "10M-50M",
            "50M-100M",
            "100M+",
        ],
    )

    yearly_trends = (
        deals_df.groupby("year")
        .agg({"id": "count", "amount": ["sum", "mean", "median", "std"]})
        .round(2)
    )

    yoy_growth = yearly_trends[("amount", "sum")].pct_change() * 100
    yearly_trends[("amount", "yoy_growth")] = yoy_growth.round(2)

    deals_df["quarter"] = pd.to_datetime(deals_df["date"]).dt.quarter
    quarterly_trends = (
        deals_df.groupby(["year", "quarter"])
        .agg({"id": "count", "amount": ["sum", "mean"]})
        .round(2)
    )

    deal_size_trends = (
        deals_df.groupby(["year", "deal_size_category"])
        .agg({"id": "count", "amount": ["sum", "mean"]})
        .round(2)
    )

    deal_size_dist = (
        pd.crosstab(deals_df["year"], deals_df["deal_size_category"], normalize="index")
        * 100
    )

    return {
        "yearly_trends": yearly_trends,
        "quarterly_trends": quarterly_trends,
        "deal_size_trends": deal_size_trends,
        "deal_size_distribution": deal_size_dist,
    }


def analyze_funding_stages(deals_df):
    """Analyze funding stages in detail"""
    stage_analysis = (
        deals_df.groupby("roundType")
        .agg({"id": "count", "amount": ["mean", "median", "sum"]})
        .round(2)
    )

    stage_evolution = (
        deals_df.groupby(["year", "roundType"])
        .agg({"id": "count", "amount": ["mean", "sum"]})
        .round(2)
    )

    return {"stage_analysis": stage_analysis, "stage_evolution": stage_evolution}


def analyze_investor_demographics(investors_df, deal_investor_df):
    """Analyze investor demographics and behavior"""
    investor_deals = deal_investor_df.merge(
        investors_df[["id", "country", "investorType"]],
        left_on="investorId",
        right_on="id",
        suffixes=("_deal", "_investor"),
    )

    country_analysis = (
        investor_deals.groupby("country")
        .agg({"dealId": "count", "leadInvestorFlag": "sum"})
        .round(2)
    )

    stage_by_country = (
        investor_deals.groupby(["country", "roundType"]).size().unstack(fill_value=0)
    )

    active_investors = (
        investor_deals.groupby(["investorName", "country"])
        .agg({"dealId": "count", "leadInvestorFlag": "sum"})
        .sort_values("dealId", ascending=False)
    )

    return {
        "country_analysis": country_analysis,
        "stage_by_country": stage_by_country,
        "active_investors": active_investors,
    }


def analyze_sectoral_regional_insights(companies_df, deals_df):
    """Analyze sectoral and regional patterns"""
    sector_funding = deals_df.merge(
        companies_df[["id", "primaryTag", "ecosystemName"]],
        left_on="companyId",
        right_on="id",
        how="left",
        suffixes=("_deal", "_company"),
    )

    print("Available columns after merge:", sector_funding.columns.tolist())

    sector_funding["primaryTag"] = sector_funding["primaryTag_company"]
    sector_funding["ecosystemName"] = sector_funding["ecosystemName_company"]

    sector_analysis = (
        sector_funding.groupby("primaryTag")
        .agg({"amount": ["sum", "mean", "count"]})
        .round(2)
    )

    regional_analysis = (
        sector_funding.groupby("ecosystemName")
        .agg({"amount": ["sum", "mean", "count"]})
        .round(2)
    )

    sector_by_region = pd.crosstab(
        sector_funding["ecosystemName"],
        sector_funding["primaryTag"],
        values=sector_funding["amount"],
        aggfunc="sum",
    ).fillna(0)

    return {
        "sector_analysis": sector_analysis,
        "regional_analysis": regional_analysis,
        "sector_by_region": sector_by_region,
    }


def get_geographic_distribution(companies_df, ecosystems_df):
    """Analyze geographic distribution of startups"""
    geo_distribution = (
        companies_df.merge(
            ecosystems_df[["ecosystemName", "city", "province"]],
            on="ecosystemName",
            how="left",
        )
        .groupby(["province", "city"])
        .agg(
            {
                "id": "count",
                "primaryTag": lambda x: list(x.value_counts().head(3).index),
            }
        )
        .reset_index()
    )

    return geo_distribution


def perform_full_eda():
    """Perform complete exploratory data analysis"""
    results = {"raw_data": {}, "error": None}

    try:
        companies_df, ecosystems_df = load_data()
        results["raw_data"]["companies"] = companies_df
        results["raw_data"]["ecosystems"] = ecosystems_df

        deals_df = pd.read_csv("runQL/data/deals.csv")
        investors_df = pd.read_csv("runQL/data/investors.csv")
        deal_investor_df = pd.read_csv("runQL/data/dealInvestor.csv")

        results["raw_data"].update(
            {
                "deals": deals_df,
                "investors": investors_df,
                "deal_investor": deal_investor_df,
            }
        )

        deals_df["date"] = pd.to_datetime(deals_df["date"])

        required_company_cols = [
            "id",
            "companyName",
            "primaryTag",
            "ecosystemName",
            "dateFounded",
            "latestRoundType",
        ]
        required_deal_cols = [
            "id",
            "companyId",
            "companyName",
            "amount",
            "roundType",
            "date",
            "year",
        ]

        missing_company_cols = [
            col for col in required_company_cols if col not in companies_df.columns
        ]
        missing_deal_cols = [
            col for col in required_deal_cols if col not in deals_df.columns
        ]

        if missing_company_cols or missing_deal_cols:
            error_msg = []
            if missing_company_cols:
                error_msg.append(
                    f"Missing columns in companies_df: {missing_company_cols}"
                )
            if missing_deal_cols:
                error_msg.append(f"Missing columns in deals_df: {missing_deal_cols}")
            raise ValueError("\n".join(error_msg))

        deals_df["amount"] = pd.to_numeric(deals_df["amount"], errors="coerce")

        try:
            results["ecosystem_stats"] = get_ecosystem_stats(companies_df)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Error in ecosystem_stats analysis: {str(e)}")

        try:
            results["sector_stats"] = get_sector_analysis(companies_df)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Error in sector_stats analysis: {str(e)}")

        try:
            results["funding_analysis"] = get_funding_analysis(companies_df)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Error in funding_analysis: {str(e)}")

        try:
            results["geographic_distribution"] = get_geographic_distribution(
                companies_df, ecosystems_df
            )
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Error in geographic_distribution analysis: {str(e)}")

        try:
            results["investment_trends"] = analyze_investment_trends(deals_df)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Error in investment_trends analysis: {str(e)}")

        try:
            results["funding_stages"] = analyze_funding_stages(deals_df)
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Error in funding_stages analysis: {str(e)}")

        try:
            results["investor_demographics"] = analyze_investor_demographics(
                investors_df, deal_investor_df
            )
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Error in investor_demographics analysis: {str(e)}")

        try:
            results["sectoral_regional"] = analyze_sectoral_regional_insights(
                companies_df, deals_df
            )
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Error in sectoral_regional analysis: {str(e)}")

        return results
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error in perform_full_eda: {str(e)}")
        results["error"] = str(e)
        return results


if __name__ == "__main__":
    results_ = perform_full_eda()

    if "error" in results_:
        print(f"Analysis completed with errors: {results_['error']}")
        insights = {"timestamp": datetime.now().isoformat(), "error": results_["error"]}

        if results_["raw_data"]["companies"] is not None:
            insights["ecosystem_summary"] = {
                "total_companies": len(results_["raw_data"]["companies"]),
                "total_ecosystems": len(
                    results_["raw_data"]["companies"]["ecosystemName"].unique()
                ),
                "avg_company_age": (
                    float(results_["raw_data"]["companies"]["age"].mean())
                    if "age" in results_["raw_data"]["companies"].columns
                    else None
                ),
            }
    else:
        insights = {
            "timestamp": datetime.now().isoformat(),
            "ecosystem_summary": {
                "total_companies": len(results_["raw_data"]["companies"]),
                "total_ecosystems": len(results_["raw_data"]["ecosystems"]),
                "avg_company_age": float(
                    results_["raw_data"]["companies"]["age"].mean()
                ),
            },
            "investment_trends": {
                "yearly_summary": results_["investment_trends"][
                    "yearly_trends"
                ].to_dict(),
                "quarterly_summary": results_["investment_trends"][
                    "quarterly_trends"
                ].to_dict(),
                "deal_size_distribution": results_["investment_trends"][
                    "deal_size_distribution"
                ].to_dict(),
            },
            "funding_stages": {
                "stage_metrics": results_["funding_stages"]["stage_analysis"].to_dict(),
                "stage_evolution": results_["funding_stages"][
                    "stage_evolution"
                ].to_dict(),
            },
            "investor_insights": {
                "country_distribution": results_["investor_demographics"][
                    "country_analysis"
                ].to_dict(),
                "stage_preferences": results_["investor_demographics"][
                    "stage_by_country"
                ].to_dict(),
                "top_investors": results_["investor_demographics"]["active_investors"]
                .head(20)
                .to_dict(),
            },
            "sector_insights": {
                "sector_metrics": results_["sectoral_regional"][
                    "sector_analysis"
                ].to_dict(),
                "regional_metrics": results_["sectoral_regional"][
                    "regional_analysis"
                ].to_dict(),
                "sector_region_matrix": results_["sectoral_regional"][
                    "sector_by_region"
                ].to_dict(),
            },
            "ecosystem_stats": results_["ecosystem_stats"].to_dict(orient="records"),
            "geographic_insights": results_["geographic_distribution"].to_dict(
                orient="records"
            ),
        }

    output_file = (
        f'runQL/data/startup_insights_{datetime.now().strftime("%Y%m%d")}.json'
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=4)

    print(f"Enhanced insights saved to {output_file}")
