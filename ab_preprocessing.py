#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

REAL_FILE = r"D:\file\Columbia\G5243\ga4_real_data.xlsx"
USER_FILE = r"D:\file\Columbia\G5243\user_level_simulated.xlsx"

REAL_SHEET = "ga4_real_data"
USER_SHEET = "user_level_simulated"


# ============================================================
# BASIC UTILS
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def check_file_exists(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def proportion_ci_wald(p, n, z=1.96):
    if n == 0 or pd.isna(p):
        return np.nan, np.nan
    se = np.sqrt(p * (1 - p) / n)
    lo = max(0, p - z * se)
    hi = min(1, p + z * se)
    return lo, hi


# ============================================================
# LOADERS
# ============================================================

def load_real_ga4_excel(filepath: str, sheet_name: str) -> pd.DataFrame:
    check_file_exists(filepath)
    raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    return raw


def load_user_level_excel(filepath: str, sheet_name: str) -> pd.DataFrame:
    check_file_exists(filepath)
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df


# ============================================================
# PART 1: REAL GA4 AGGREGATED DATA PREPROCESSING
# ============================================================

def preprocess_real_ga4(raw_df: pd.DataFrame):
    print_section("PART 1 — PREPROCESSING REAL GA4 AGGREGATED DATA")

    print(f"Raw sheet shape: {raw_df.shape}")

    # --------------------------------------------------------
    # 1. Find header row
    # --------------------------------------------------------
    header_row_idx = None
    for i in range(len(raw_df)):
        vals = raw_df.iloc[i].astype(str).tolist()
        if "ab_version" in vals:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError("Could not find header row containing 'ab_version' in the real GA4 sheet.")

    print(f"Detected header row index: {header_row_idx}")

    # --------------------------------------------------------
    # 2. Slice actual data table
    # --------------------------------------------------------
    data_start = header_row_idx + 2
    df = raw_df.iloc[data_start:].copy().reset_index(drop=True)
    df = df.iloc[:, :5].copy()
    df.columns = ["event", "not_set", "A", "B", "total"]

    # --------------------------------------------------------
    # 3. Clean rows
    # --------------------------------------------------------
    df = df[~df["event"].isna()].copy()
    df["event"] = df["event"].astype(str).str.strip()

    df = df[df["event"] != ""]
    df = df[df["event"].str.lower() != "nan"]
    df = df[~df["event"].str.contains("total", case=False, na=False)]

    # --------------------------------------------------------
    # 4. Numeric conversion
    # --------------------------------------------------------
    for col in ["not_set", "A", "B", "total"]:
        df[col] = safe_numeric(df[col]).fillna(0).astype(int)

    # --------------------------------------------------------
    # 5. Checks
    # --------------------------------------------------------
    df["sum_check"] = df["not_set"] + df["A"] + df["B"]
    df["total_match"] = (df["sum_check"] == df["total"]).astype(int)

    df["not_set_rate"] = np.where(df["total"] > 0, df["not_set"] / df["total"], np.nan)

    key_events = [
        "button_click",
        "tab_switch",
        "tab_duration",
        "page_view",
        "session_start",
        "scroll",
        "first_visit",
        "session_duration",
        "ab_assignment",
        "user_engagement"
    ]
    df["is_key_event"] = df["event"].isin(key_events).astype(int)

    print("\n[Cleaned Aggregated Table]")
    print(df.to_string(index=False))

    print("\n[Aggregated Data Quality Summary]")
    print(f"Rows in raw sheet                : {len(raw_df)}")
    print(f"Rows in cleaned event table      : {len(df)}")
    print(f"Rows with total match            : {int(df['total_match'].sum())}")
    print(f"Rows with total mismatch         : {int((df['total_match'] == 0).sum())}")
    print(f"Key events found                 : {int(df['is_key_event'].sum())}")

    print("\n[(not set) Share by Event]")
    print(df[["event", "not_set", "total", "not_set_rate"]].to_string(index=False))

    # --------------------------------------------------------
    # 6. Session proxy: prioritize session_start
    # --------------------------------------------------------
    lookup = {row["event"]: row for _, row in df.iterrows()}

    if "session_start" in lookup:
        n_A = int(lookup["session_start"]["A"])
        n_B = int(lookup["session_start"]["B"])
        session_proxy = "session_start"
    elif "session_duration" in lookup:
        n_A = int(lookup["session_duration"]["A"])
        n_B = int(lookup["session_duration"]["B"])
        session_proxy = "session_duration"
    else:
        n_A = 0
        n_B = 0
        session_proxy = "not found"

    total_A_events = int(df["A"].sum())
    total_B_events = int(df["B"].sum())

    # --------------------------------------------------------
    # 7. Derived descriptive metrics
    # --------------------------------------------------------
    derived_rows = []
    for _, row in df.iterrows():
        a_rate = row["A"] / n_A if n_A > 0 else np.nan
        b_rate = row["B"] / n_B if n_B > 0 else np.nan

        a_share = row["A"] / total_A_events if total_A_events > 0 else np.nan
        b_share = row["B"] / total_B_events if total_B_events > 0 else np.nan

        derived_rows.append({
            "event": row["event"],
            "A_count": row["A"],
            "B_count": row["B"],
            "not_set_count": row["not_set"],
            "not_set_rate": row["not_set_rate"],
            "A_per_session": a_rate,
            "B_per_session": b_rate,
            "per_session_diff_B_minus_A": b_rate - a_rate if pd.notna(a_rate) and pd.notna(b_rate) else np.nan,
            "A_event_share": a_share,
            "B_event_share": b_share,
        })

    derived_df = pd.DataFrame(derived_rows)

    print("\n[Derived Aggregated Descriptive Metrics]")
    print(f"Session proxy used               : {session_proxy}")
    print(f"Estimated sessions A             : {n_A}")
    print(f"Estimated sessions B             : {n_B}")
    print(f"Total A tagged events            : {total_A_events}")
    print(f"Total B tagged events            : {total_B_events}")
    print(derived_df.to_string(index=False))

    # --------------------------------------------------------
    # 8. Lightweight warnings
    # --------------------------------------------------------
    print("\n[Part 1 Warnings / Notes]")
    high_not_set = derived_df[derived_df["not_set_rate"] > 0.15]
    if len(high_not_set) > 0:
        print("The following events have relatively high '(not set)' share:")
        print(high_not_set[["event", "not_set_rate"]].to_string(index=False))
    else:
        print("No event has a very high '(not set)' share by current threshold.")

    ab_row = derived_df[derived_df["event"] == "ab_assignment"]
    if len(ab_row) > 0:
        a_ct = int(ab_row.iloc[0]["A_count"])
        b_ct = int(ab_row.iloc[0]["B_count"])
        if a_ct == 0 and b_ct == 0:
            print("ab_assignment has no tagged A/B counts and will not be emphasized in plots.")

    return df, derived_df, {
        "session_proxy": session_proxy,
        "n_A": n_A,
        "n_B": n_B,
        "total_A_events": total_A_events,
        "total_B_events": total_B_events
    }


# ============================================================
# PART 1: REAL GA4 VISUALIZATION
# ============================================================

def plot_real_ga4_overview(derived_df: pd.DataFrame):
    print_section("PART 1 — VISUALIZATION REAL GA4 AGGREGATED DATA")

    # Remove events that are not useful for comparison charts
    plot_df = derived_df[derived_df["event"].isin([
        "button_click", "tab_switch", "tab_duration",
        "scroll", "session_duration", "user_engagement"
    ])].copy()

    if len(plot_df) == 0:
        print("No plottable key events found in aggregated GA4 data.")
        return

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, plot_df["A_per_session"], width, label="A")
    plt.bar(x + width / 2, plot_df["B_per_session"], width, label="B")
    plt.xticks(x, plot_df["event"], rotation=30)
    plt.ylabel("Per-session event rate")
    plt.title("Part 1: Real GA4 Aggregated Data — Event Rate per Session")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, plot_df["A_event_share"], width, label="A")
    plt.bar(x + width / 2, plot_df["B_event_share"], width, label="B")
    plt.xticks(x, plot_df["event"], rotation=30)
    plt.ylabel("Share of tagged A/B events")
    plt.title("Part 1: Real GA4 Aggregated Data — Event Share by Version")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# PART 2: USER-LEVEL PREPROCESSING
# ============================================================

def preprocess_user_level(df_raw: pd.DataFrame):
    print_section("PART 2 — PREPROCESSING USER-LEVEL DATA")

    df = df_raw.copy()
    print(f"Raw user-level shape: {df.shape}")

    # --------------------------------------------------------
    # 1. Standardize column names
    # --------------------------------------------------------
    df.columns = [c.strip().lower() for c in df.columns]
    print("\n[Columns after standardization]")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # 2. Required columns
    # --------------------------------------------------------
    required_cols = [
        "user_id",
        "ab_version",
        "tab_switches",
        "button_clicks",
        "guided_clicks",
        "scroll_count",
        "total_tab_duration_sec",
        "avg_tab_duration_sec",
        "session_duration_sec",
        "unique_tabs_visited",
        "reached_cleaning",
        "reached_feature_eng",
        "reached_eda",
        "workflow_depth",
        "linear_path_score",
        "bounced",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in user-level file: {missing_cols}")

    # --------------------------------------------------------
    # 3. Remove duplicate users
    # --------------------------------------------------------
    dup_count = int(df["user_id"].duplicated().sum())
    df = df.drop_duplicates(subset=["user_id"], keep="first").copy()

    # --------------------------------------------------------
    # 4. Clean labels
    # --------------------------------------------------------
    df["ab_version"] = df["ab_version"].astype(str).str.strip().str.upper()
    df["missing_ab_version"] = (~df["ab_version"].isin(["A", "B"])).astype(int)

    # --------------------------------------------------------
    # 5. Numeric conversion
    # --------------------------------------------------------
    continuous_cols = [
        "tab_switches",
        "button_clicks",
        "guided_clicks",
        "scroll_count",
        "total_tab_duration_sec",
        "avg_tab_duration_sec",
        "session_duration_sec",
        "unique_tabs_visited",
        "workflow_depth",
        "linear_path_score",
    ]
    binary_cols = [
        "reached_cleaning",
        "reached_feature_eng",
        "reached_eda",
        "bounced",
    ]

    for col in continuous_cols:
        df[col] = safe_numeric(df[col])

    for col in binary_cols:
        df[col] = safe_numeric(df[col]).fillna(0).clip(0, 1).astype(int)

    print("\n[Missing Value Count]")
    print(df[required_cols].isna().sum().to_string())

    # --------------------------------------------------------
    # 6. Validity flags
    # --------------------------------------------------------
    df["too_short_session"] = (df["session_duration_sec"] < 2).fillna(False).astype(int)
    df["invalid_negative_session_duration"] = (df["session_duration_sec"] < 0).fillna(False).astype(int)
    df["invalid_negative_total_tab_duration"] = (df["total_tab_duration_sec"] < 0).fillna(False).astype(int)
    df["invalid_negative_avg_tab_duration"] = (df["avg_tab_duration_sec"] < 0).fillna(False).astype(int)
    df["invalid_negative_clicks"] = (
        (df["button_clicks"] < 0) |
        (df["guided_clicks"] < 0) |
        (df["scroll_count"] < 0) |
        (df["tab_switches"] < 0)
    ).fillna(False).astype(int)

    df["valid_session"] = (
        (df["missing_ab_version"] == 0) &
        (df["too_short_session"] == 0) &
        (df["invalid_negative_session_duration"] == 0) &
        (df["invalid_negative_total_tab_duration"] == 0) &
        (df["invalid_negative_avg_tab_duration"] == 0) &
        (df["invalid_negative_clicks"] == 0)
    ).astype(int)

    before_filter = len(df)
    df = df[df["valid_session"] == 1].copy()
    after_filter = len(df)

    # --------------------------------------------------------
    # 7. Derived metrics
    # --------------------------------------------------------
    df["deep_engagement"] = (df["workflow_depth"] >= 2).astype(int)

    df["completed_workflow"] = (
        (df["reached_cleaning"] == 1) &
        ((df["reached_feature_eng"] == 1) | (df["reached_eda"] == 1))
    ).astype(int)

    # IMPORTANT: rename from share -> intensity
    # because in this dataset guided_clicks may exceed button_clicks
    df["guided_click_intensity"] = np.where(
        df["button_clicks"] > 0,
        df["guided_clicks"] / df["button_clicks"],
        0
    )

    df["duration_per_tab"] = np.where(
        df["unique_tabs_visited"] > 0,
        df["total_tab_duration_sec"] / df["unique_tabs_visited"],
        np.nan
    )

    df["scrolls_per_tab"] = np.where(
        df["unique_tabs_visited"] > 0,
        df["scroll_count"] / df["unique_tabs_visited"],
        np.nan
    )

    df["clicks_per_tab"] = np.where(
        df["unique_tabs_visited"] > 0,
        df["button_clicks"] / df["unique_tabs_visited"],
        np.nan
    )

    df["guided_clicks_per_tab"] = np.where(
        df["unique_tabs_visited"] > 0,
        df["guided_clicks"] / df["unique_tabs_visited"],
        np.nan
    )

    df["workflow_stage_sum"] = (
        df["reached_cleaning"] +
        df["reached_feature_eng"] +
        df["reached_eda"]
    )
    df["workflow_depth_match"] = (df["workflow_depth"] == df["workflow_stage_sum"]).astype(int)

    # --------------------------------------------------------
    # 8. Outlier caps
    # --------------------------------------------------------
    cap_cols = [
        "tab_switches",
        "button_clicks",
        "guided_clicks",
        "scroll_count",
        "total_tab_duration_sec",
        "avg_tab_duration_sec",
        "session_duration_sec",
        "unique_tabs_visited",
        "workflow_depth",
        "linear_path_score",
        "guided_click_intensity",
        "duration_per_tab",
        "scrolls_per_tab",
        "clicks_per_tab",
        "guided_clicks_per_tab",
    ]

    cap_summary = []
    for col in cap_cols:
        if df[col].notna().sum() == 0:
            continue
        cap = float(df[col].quantile(0.99))
        df[f"{col}_clean"] = df[col].clip(upper=cap)
        cap_summary.append((col, round(cap, 4)))

    # --------------------------------------------------------
    # 9. Data quality summary
    # --------------------------------------------------------
    print("\n[Data Quality Summary]")
    print(f"raw_rows                              : {len(df_raw)}")
    print(f"duplicate_user_ids_removed            : {dup_count}")
    print(f"rows_before_validity_filter           : {before_filter}")
    print(f"rows_after_validity_filter            : {after_filter}")
    print(f"missing_ab_version_in_raw             : {int((~df_raw['ab_version'].astype(str).str.strip().str.upper().isin(['A','B'])).sum())}")
    print(f"too_short_session_in_raw              : {int((safe_numeric(df_raw['session_duration_sec']) < 2).sum())}")
    print(f"workflow_depth_mismatch_count         : {int((df['workflow_depth_match'] == 0).sum())}")
    print(f"group_A_n                             : {int((df['ab_version'] == 'A').sum())}")
    print(f"group_B_n                             : {int((df['ab_version'] == 'B').sum())}")

    print("\n[99th Percentile Caps]")
    for col, cap in cap_summary:
        print(f"{col:<35}: {cap}")

    # --------------------------------------------------------
    # 10. Metric dictionary
    # --------------------------------------------------------
    metric_dict = pd.DataFrame([
        ["deep_engagement", "Primary", "1 if workflow_depth >= 2"],
        ["completed_workflow", "Secondary", "1 if Cleaning reached and then Feature Eng or EDA reached"],
        ["guided_click_intensity", "Secondary", "guided_clicks / button_clicks; intensity, not bounded share"],
        ["duration_per_tab", "Secondary", "total_tab_duration_sec / unique_tabs_visited"],
        ["scrolls_per_tab", "Secondary", "scroll_count / unique_tabs_visited"],
        ["clicks_per_tab", "Secondary", "button_clicks / unique_tabs_visited"],
        ["guided_clicks_per_tab", "Secondary", "guided_clicks / unique_tabs_visited"],
        ["bounced", "Sanity", "1 if user bounced"],
        ["valid_session", "Sanity", "1 if row passes inclusion rules"],
        ["workflow_depth_match", "Sanity", "1 if workflow_depth equals stage-sum"],
    ], columns=["metric_name", "role", "definition"])

    print("\n[Metric Dictionary]")
    print(metric_dict.to_string(index=False))

    # --------------------------------------------------------
    # 11. Split summaries into 3 readable tables
    # --------------------------------------------------------
    core_rows = []
    funnel_rows = []
    style_rows = []

    for version in ["A", "B"]:
        g = df[df["ab_version"] == version].copy()
        n = len(g)

        de = g["deep_engagement"].mean()
        cw = g["completed_workflow"].mean()
        br = g["bounced"].mean()

        de_lo, de_hi = proportion_ci_wald(de, n)
        cw_lo, cw_hi = proportion_ci_wald(cw, n)
        br_lo, br_hi = proportion_ci_wald(br, n)

        core_rows.append({
            "ab_version": version,
            "n": n,
            "deep_engagement_rate": round(de, 4),
            "deep_engagement_ci_low": round(de_lo, 4) if pd.notna(de_lo) else np.nan,
            "deep_engagement_ci_high": round(de_hi, 4) if pd.notna(de_hi) else np.nan,
            "completed_workflow_rate": round(cw, 4),
            "completed_workflow_ci_low": round(cw_lo, 4) if pd.notna(cw_lo) else np.nan,
            "completed_workflow_ci_high": round(cw_hi, 4) if pd.notna(cw_hi) else np.nan,
            "mean_workflow_depth": round(g["workflow_depth_clean"].mean(), 4),
            "mean_linear_path_score": round(g["linear_path_score_clean"].mean(), 4),
            "mean_session_duration_sec": round(g["session_duration_sec_clean"].mean(), 4),
        })

        funnel_rows.append({
            "ab_version": version,
            "reached_cleaning_rate": round(g["reached_cleaning"].mean(), 4),
            "reached_feature_eng_rate": round(g["reached_feature_eng"].mean(), 4),
            "reached_eda_rate": round(g["reached_eda"].mean(), 4),
        })

        style_rows.append({
            "ab_version": version,
            "mean_tab_switches": round(g["tab_switches_clean"].mean(), 4),
            "mean_button_clicks": round(g["button_clicks_clean"].mean(), 4),
            "mean_guided_clicks": round(g["guided_clicks_clean"].mean(), 4),
            "mean_guided_click_intensity": round(g["guided_click_intensity_clean"].mean(), 4),
            "mean_unique_tabs": round(g["unique_tabs_visited_clean"].mean(), 4),
            "bounce_rate": round(br, 4),
            "bounce_ci_low": round(br_lo, 4) if pd.notna(br_lo) else np.nan,
            "bounce_ci_high": round(br_hi, 4) if pd.notna(br_hi) else np.nan,
        })

    core_df = pd.DataFrame(core_rows)
    funnel_df = pd.DataFrame(funnel_rows)
    style_df = pd.DataFrame(style_rows)

    print("\n[Core Results Summary]")
    print(core_df.to_string(index=False))

    print("\n[Funnel Summary]")
    print(funnel_df.to_string(index=False))

    print("\n[Behavior / Style Summary]")
    print(style_df.to_string(index=False))

    # --------------------------------------------------------
    # 12. Preview
    # --------------------------------------------------------
    preview_cols = [
        "user_id", "ab_version",
        "tab_switches", "tab_switches_clean",
        "button_clicks", "button_clicks_clean",
        "guided_clicks", "guided_clicks_clean",
        "guided_click_intensity", "guided_click_intensity_clean",
        "session_duration_sec", "session_duration_sec_clean",
        "workflow_depth", "workflow_depth_clean",
        "deep_engagement", "completed_workflow",
        "bounced", "valid_session"
    ]

    print("\n[Preview of Cleaned User-Level Table]")
    print(df[preview_cols].head(12).to_string(index=False))

    return df, core_df, funnel_df, style_df, metric_dict


# ============================================================
# PART 2: USER-LEVEL VISUALIZATION
# ============================================================

def plot_user_level_sample_balance(df: pd.DataFrame):
    counts = df["ab_version"].value_counts().reindex(["A", "B"]).fillna(0)

    plt.figure(figsize=(6, 4))
    ax = counts.plot(kind="bar")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom")
    plt.title("Part 2: Sample Size by A/B Version")
    plt.xlabel("Version")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.show()


def plot_user_level_primary_metric(df: pd.DataFrame):
    rates = df.groupby("ab_version")["deep_engagement"].mean().reindex(["A", "B"]).fillna(0) * 100

    plt.figure(figsize=(6, 4))
    ax = rates.plot(kind="bar")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom")
    plt.title("Part 2: Primary Metric — Deep Engagement Rate")
    plt.xlabel("Version")
    plt.ylabel("Rate (%)")
    plt.tight_layout()
    plt.show()


def plot_user_level_completed_workflow(df: pd.DataFrame):
    rates = df.groupby("ab_version")["completed_workflow"].mean().reindex(["A", "B"]).fillna(0) * 100

    plt.figure(figsize=(6, 4))
    ax = rates.plot(kind="bar")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom")
    plt.title("Part 2: Completed Workflow Rate")
    plt.xlabel("Version")
    plt.ylabel("Rate (%)")
    plt.tight_layout()
    plt.show()


def plot_user_level_funnel(df: pd.DataFrame):
    stages = ["reached_cleaning", "reached_feature_eng", "reached_eda"]
    labels = ["Cleaning", "Feature Eng", "EDA"]

    a_rates = [df[df["ab_version"] == "A"][col].mean() * 100 for col in stages]
    b_rates = [df[df["ab_version"] == "B"][col].mean() * 100 for col in stages]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, a_rates, width, label="A")
    plt.bar(x + width / 2, b_rates, width, label="B")
    plt.xticks(x, labels)
    plt.ylabel("Rate (%)")
    plt.title("Part 2: Workflow Funnel by A/B Version")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_user_level_duration(df: pd.DataFrame):
    a = df[df["ab_version"] == "A"]["session_duration_sec_clean"].dropna()
    b = df[df["ab_version"] == "B"]["session_duration_sec_clean"].dropna()

    plt.figure(figsize=(7, 5))
    plt.boxplot([a, b], tick_labels=["A", "B"])
    plt.ylabel("Session Duration (sec)")
    plt.title("Part 2: Session Duration Distribution")
    plt.tight_layout()
    plt.show()


def plot_user_level_clicks(df: pd.DataFrame):
    a = df[df["ab_version"] == "A"]["button_clicks_clean"].dropna()
    b = df[df["ab_version"] == "B"]["button_clicks_clean"].dropna()

    plt.figure(figsize=(7, 5))
    plt.boxplot([a, b], tick_labels=["A", "B"])
    plt.ylabel("Button Clicks")
    plt.title("Part 2: Button Click Distribution")
    plt.tight_layout()
    plt.show()


def plot_user_level_guided_click_intensity(df: pd.DataFrame):
    a = df[df["ab_version"] == "A"]["guided_click_intensity_clean"].dropna()
    b = df[df["ab_version"] == "B"]["guided_click_intensity_clean"].dropna()

    plt.figure(figsize=(7, 5))
    plt.boxplot([a, b], tick_labels=["A", "B"])
    plt.ylabel("Guided Click Intensity")
    plt.title("Part 2: Guided Click Intensity Distribution")
    plt.tight_layout()
    plt.show()


def plot_user_level_linear_path_score(df: pd.DataFrame):
    a = df[df["ab_version"] == "A"]["linear_path_score_clean"].dropna()
    b = df[df["ab_version"] == "B"]["linear_path_score_clean"].dropna()

    plt.figure(figsize=(7, 5))
    plt.boxplot([a, b], tick_labels=["A", "B"])
    plt.ylabel("Linear Path Score")
    plt.title("Part 2: Linear Path Score Distribution")
    plt.tight_layout()
    plt.show()


def generate_user_level_plots(df: pd.DataFrame):
    print_section("PART 2 — VISUALIZATION USER-LEVEL DATA")
    print("Showing plots: sample balance, deep engagement, completed workflow, funnel,")
    print("duration, button clicks, guided click intensity, linear path score")
    print("Bounce is not plotted because it is 0% in both groups and not informative in this dataset.")

    plot_user_level_sample_balance(df)
    plot_user_level_primary_metric(df)
    plot_user_level_completed_workflow(df)
    plot_user_level_funnel(df)
    plot_user_level_duration(df)
    plot_user_level_clicks(df)
    plot_user_level_guided_click_intensity(df)
    plot_user_level_linear_path_score(df)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print_section("STAT 5243 Project 3 — IMPROVED INPUT + PREPROCESSING + VISUALIZATION")

    # --------------------------------------------------------
    # Part 1: Real GA4 aggregated data
    # --------------------------------------------------------
    real_raw = load_real_ga4_excel(REAL_FILE, REAL_SHEET)
    real_clean_df, real_derived_df, real_meta = preprocess_real_ga4(real_raw)
    plot_real_ga4_overview(real_derived_df)

    # --------------------------------------------------------
    # Part 2: User-level data
    # --------------------------------------------------------
    user_raw = load_user_level_excel(USER_FILE, USER_SHEET)
    user_clean_df, core_df, funnel_df, style_df, metric_dict = preprocess_user_level(user_raw)
    generate_user_level_plots(user_clean_df)

    print_section("DONE")
    print("All preprocessing, summaries, and plots have been displayed.")
    print("No files were saved to disk.")


# In[ ]:




