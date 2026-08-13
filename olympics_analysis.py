"""
🏅 Olympic Games EDA — Python Data Analysis Project
=====================================================
Dataset  : 5000 Olympic Medal Records (1896-2024)
Libraries: Pandas, Matplotlib, Seaborn, NumPy
Charts   : 10 Publication-ready visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "olympics_medals.csv")
CHARTS = os.path.join(BASE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────
GOLD_C   = "#FFD700"
SILVER_C = "#C0C0C0"
BRONZE_C = "#CD7F32"
PALETTE  = ["#003580","#FFD700","#C0C0C0","#CD7F32","#E63946","#2D6A4F","#F4A261","#264653","#E9C46A","#A8DADC"]
BG       = "#0A0A1A"
TEXT     = "#FFFFFF"
GRID     = "#2A2A3A"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   GOLD_C,
})

def save(name):
    path = os.path.join(CHARTS, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✅ {name}")

# ══════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════
print("\n🏅 Loading Olympics data...")
df = pd.read_csv(DATA)

print(f"  Shape       : {df.shape}")
print(f"  Years       : {df['year'].min()} — {df['year'].max()}")
print(f"  Sports      : {df['sport'].nunique()}")
print(f"  Countries   : {df['country'].nunique()}")
print(f"  Athletes    : {df['athlete_name'].nunique()}")
print(f"  Total Gold  : {(df['medal']=='Gold').sum()}")

# Feature Engineering
df["era"] = pd.cut(df["year"],
    bins=[1895,1920,1945,1970,1995,2024],
    labels=["Early Era\n(1896-1920)","Inter-War\n(1921-1945)",
            "Cold War\n(1946-1970)","Modern Era\n(1971-1995)","Contemporary\n(1996-2024)"])

df["medal_points"] = df["medal"].map({"Gold":3,"Silver":2,"Bronze":1})

print("\n📊 Generating charts...")

# ══════════════════════════════════════════════════════════════
# 2. CHART 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("🏅 Olympic Games EDA — Overview Dashboard (1896-2024)",
             fontsize=16, fontweight="bold", y=1.01, color=GOLD_C)
fig.patch.set_facecolor(BG)

# 1a. Medal Distribution
medal_counts = df["medal"].value_counts()
colors_medal = [GOLD_C, SILVER_C, BRONZE_C]
axes[0,0].pie(medal_counts.values, labels=medal_counts.index,
              autopct="%1.1f%%", colors=colors_medal, startangle=140,
              wedgeprops={"edgecolor": BG, "linewidth": 2})
axes[0,0].set_facecolor(BG)
axes[0,0].set_title("Medal Distribution", color=GOLD_C)

# 1b. Summer vs Winter
season_counts = df["season"].value_counts()
axes[0,1].bar(season_counts.index, season_counts.values,
              color=["#FF6200","#00B4D8"], edgecolor=BG, width=0.5)
axes[0,1].set_facecolor(BG)
axes[0,1].set_title("Summer vs Winter Olympics", color=GOLD_C)
axes[0,1].set_ylabel("Medal Count", color=TEXT)
for bar in axes[0,1].patches:
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                   str(int(bar.get_height())), ha="center", fontsize=10, color=TEXT)

# 1c. Gender Distribution
gender_counts = df["gender"].value_counts()
axes[0,2].pie(gender_counts.values, labels=["Male","Female"],
              autopct="%1.1f%%", colors=["#003580","#E63946"],
              startangle=140, wedgeprops={"edgecolor": BG, "linewidth": 2})
axes[0,2].set_facecolor(BG)
axes[0,2].set_title("Gender Distribution", color=GOLD_C)

# 1d. Top 10 Countries by Total Medals
top_countries = df["country"].value_counts().head(10).sort_values()
axes[1,0].barh(top_countries.index, top_countries.values, color=GOLD_C)
axes[1,0].set_facecolor(BG)
axes[1,0].set_title("Top 10 Countries by Total Medals", color=GOLD_C)
axes[1,0].set_xlabel("Total Medals", color=TEXT)

# 1e. Top 10 Sports by Medals
top_sports = df["sport"].value_counts().head(10).sort_values()
axes[1,1].barh(top_sports.index, top_sports.values, color="#003580")
axes[1,1].set_facecolor(BG)
axes[1,1].set_title("Top 10 Sports by Medals", color=GOLD_C)
axes[1,1].set_xlabel("Total Medals", color=TEXT)

# 1f. Age Distribution
axes[1,2].hist(df["age"], bins=20, color="#E63946", edgecolor=BG, linewidth=0.8)
axes[1,2].axvline(df["age"].mean(), color=GOLD_C, linestyle="--",
                  label=f"Mean: {df['age'].mean():.1f} yrs")
axes[1,2].set_facecolor(BG)
axes[1,2].set_title("Athlete Age Distribution", color=GOLD_C)
axes[1,2].set_xlabel("Age", color=TEXT)
axes[1,2].legend(facecolor=BG, labelcolor=TEXT)

plt.tight_layout()
save("01_overview_dashboard.png")

# ══════════════════════════════════════════════════════════════
# 3. CHART 2 — COUNTRY MEDAL ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("🌍 Country Medal Analysis", fontsize=15,
             fontweight="bold", color=GOLD_C)

# Gold Silver Bronze stacked bar - top 15 countries
top15 = df.groupby("country")["medal"].value_counts().unstack(fill_value=0)
top15["Total"] = top15.sum(axis=1)
top15 = top15.nlargest(15, "Total").drop("Total", axis=1)
top15 = top15.sort_values("Gold", ascending=True)

x = range(len(top15))
axes[0].barh(top15.index, top15.get("Gold", 0), color=GOLD_C, label="Gold")
axes[0].barh(top15.index, top15.get("Silver", 0),
             left=top15.get("Gold", 0), color=SILVER_C, label="Silver")
axes[0].barh(top15.index, top15.get("Bronze", 0),
             left=top15.get("Gold", 0)+top15.get("Silver", 0),
             color=BRONZE_C, label="Bronze")
axes[0].set_facecolor(BG)
axes[0].set_title("Top 15 Countries — Gold/Silver/Bronze", color=GOLD_C)
axes[0].set_xlabel("Medal Count", color=TEXT)
axes[0].legend(facecolor=BG, labelcolor=TEXT)

# Continent-wise medals
continent_medals = df.groupby("continent")["medal"].count().sort_values(ascending=False)
axes[1].bar(continent_medals.index, continent_medals.values,
            color=PALETTE[:len(continent_medals)], edgecolor=BG)
axes[1].set_facecolor(BG)
axes[1].set_title("Medals by Continent", color=GOLD_C)
axes[1].set_ylabel("Total Medals", color=TEXT)
axes[1].tick_params(axis="x", rotation=30)
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 str(int(bar.get_height())), ha="center", fontsize=9, color=TEXT)

plt.tight_layout()
save("02_country_medal_analysis.png")

# ══════════════════════════════════════════════════════════════
# 4. CHART 3 — HISTORICAL TREND
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("📅 Historical Medal Trends", fontsize=15,
             fontweight="bold", color=GOLD_C)

# Year-wise medals
year_medals = df.groupby("year")["medal"].count()
axes[0].plot(year_medals.index, year_medals.values,
             marker="o", color=GOLD_C, linewidth=2, markersize=5)
axes[0].fill_between(year_medals.index, year_medals.values,
                     alpha=0.15, color=GOLD_C)
axes[0].set_facecolor(BG)
axes[0].set_title("Total Medals by Year", color=GOLD_C)
axes[0].set_xlabel("Year", color=TEXT)
axes[0].set_ylabel("Total Medals", color=TEXT)

# Era-wise medals
era_medals = df.groupby("era")["medal"].count()
axes[1].bar(range(len(era_medals)), era_medals.values,
            color=PALETTE[:len(era_medals)], edgecolor=BG)
axes[1].set_facecolor(BG)
axes[1].set_title("Medals by Era", color=GOLD_C)
axes[1].set_ylabel("Total Medals", color=TEXT)
axes[1].set_xticks(range(len(era_medals)))
axes[1].set_xticklabels(era_medals.index, fontsize=8)
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                 str(int(bar.get_height())), ha="center", fontsize=9, color=TEXT)

plt.tight_layout()
save("03_historical_trends.png")

# ══════════════════════════════════════════════════════════════
# 5. CHART 4 — GENDER ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("👥 Gender Analysis in Olympics", fontsize=15,
             fontweight="bold", color=GOLD_C)

# Gender trend over years
gender_year = df.groupby(["year","gender"])["medal"].count().unstack(fill_value=0)
if "M" in gender_year.columns:
    axes[0].plot(gender_year.index, gender_year["M"], color="#003580",
                 marker="o", linewidth=2, label="Male", markersize=4)
if "F" in gender_year.columns:
    axes[0].plot(gender_year.index, gender_year["F"], color="#E63946",
                 marker="o", linewidth=2, label="Female", markersize=4)
axes[0].set_facecolor(BG)
axes[0].set_title("Male vs Female Medals Over Years", color=GOLD_C)
axes[0].set_xlabel("Year", color=TEXT)
axes[0].set_ylabel("Medals", color=TEXT)
axes[0].legend(facecolor=BG, labelcolor=TEXT)

# Age distribution by gender
male_ages = df[df["gender"]=="M"]["age"]
female_ages = df[df["gender"]=="F"]["age"]
axes[1].hist(male_ages, bins=20, alpha=0.7, color="#003580", label="Male", edgecolor=BG)
axes[1].hist(female_ages, bins=20, alpha=0.7, color="#E63946", label="Female", edgecolor=BG)
axes[1].set_facecolor(BG)
axes[1].set_title("Age Distribution by Gender", color=GOLD_C)
axes[1].set_xlabel("Age", color=TEXT)
axes[1].set_ylabel("Count", color=TEXT)
axes[1].legend(facecolor=BG, labelcolor=TEXT)

plt.tight_layout()
save("04_gender_analysis.png")

# ══════════════════════════════════════════════════════════════
# 6. CHART 5 — SPORT ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor(BG)
fig.suptitle("🏋️ Sport Analysis", fontsize=15,
             fontweight="bold", color=GOLD_C)

# Top sports by gold medals
sport_gold = df[df["medal"]=="Gold"]["sport"].value_counts().head(15).sort_values()
axes[0].barh(sport_gold.index, sport_gold.values, color=GOLD_C)
axes[0].set_facecolor(BG)
axes[0].set_title("Top 15 Sports by Gold Medals", color=GOLD_C)
axes[0].set_xlabel("Gold Medals", color=TEXT)

# Sport-wise gender split
sport_gender = df.groupby(["sport","gender"])["medal"].count().unstack(fill_value=0).head(15)
sport_gender = sport_gender.sort_values(sport_gender.columns[0] if len(sport_gender.columns)>0 else "M", ascending=True)
if "M" in sport_gender.columns and "F" in sport_gender.columns:
    axes[1].barh(sport_gender.index, sport_gender["M"], color="#003580", label="Male")
    axes[1].barh(sport_gender.index, -sport_gender["F"], color="#E63946", label="Female")
axes[1].set_facecolor(BG)
axes[1].set_title("Sport Gender Split (Top 15)", color=GOLD_C)
axes[1].set_xlabel("← Female | Male →", color=TEXT)
axes[1].legend(facecolor=BG, labelcolor=TEXT)
axes[1].axvline(0, color=TEXT, linewidth=0.5)

plt.tight_layout()
save("05_sport_analysis.png")

# ══════════════════════════════════════════════════════════════
# 7. CHART 6 — HOST COUNTRY ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("🏟️ Host City & Country Analysis", fontsize=15,
             fontweight="bold", color=GOLD_C)

# Most hosted cities
city_counts = df.groupby("city")["year"].nunique().sort_values(ascending=False).head(12)
axes[0].bar(city_counts.index, city_counts.values,
            color=PALETTE[:len(city_counts)], edgecolor=BG)
axes[0].set_facecolor(BG)
axes[0].set_title("Most Frequent Host Cities", color=GOLD_C)
axes[0].set_ylabel("Times Hosted", color=TEXT)
axes[0].tick_params(axis="x", rotation=45)

# Host country medal advantage
host_medals = df.groupby("host_country")["medal"].count().sort_values(ascending=False).head(10)
axes[1].bar(host_medals.index, host_medals.values, color=GOLD_C, edgecolor=BG)
axes[1].set_facecolor(BG)
axes[1].set_title("Medals Won by Host Countries", color=GOLD_C)
axes[1].set_ylabel("Total Medals", color=TEXT)
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
save("06_host_city_analysis.png")

# ══════════════════════════════════════════════════════════════
# 8. CHART 7 — AGE ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("👤 Athlete Age Analysis", fontsize=15,
             fontweight="bold", color=GOLD_C)

# Age by medal type
sns.boxplot(data=df, x="medal", y="age", order=["Gold","Silver","Bronze"],
            palette=[GOLD_C, SILVER_C, BRONZE_C], ax=axes[0])
axes[0].set_facecolor(BG)
axes[0].set_title("Age Distribution by Medal", color=GOLD_C)
axes[0].set_xlabel("Medal", color=TEXT)
axes[0].set_ylabel("Age", color=TEXT)

# Avg age by sport (top 12)
sport_age = df.groupby("sport")["age"].mean().sort_values(ascending=False).head(12)
axes[1].bar(sport_age.index, sport_age.values, color="#003580", edgecolor=BG)
axes[1].set_facecolor(BG)
axes[1].set_title("Avg Athlete Age by Sport (Top 12)", color=GOLD_C)
axes[1].set_ylabel("Average Age", color=TEXT)
axes[1].tick_params(axis="x", rotation=45)
axes[1].axhline(df["age"].mean(), color=GOLD_C, linestyle="--",
                label=f"Overall Avg: {df['age'].mean():.1f}")
axes[1].legend(facecolor=BG, labelcolor=TEXT)

plt.tight_layout()
save("07_age_analysis.png")

# ══════════════════════════════════════════════════════════════
# 9. CHART 8 — SUMMER vs WINTER
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("☀️❄️ Summer vs Winter Olympics", fontsize=15,
             fontweight="bold", color=GOLD_C)

# Top countries Summer
summer_top = df[df["season"]=="Summer"]["country"].value_counts().head(10).sort_values()
axes[0].barh(summer_top.index, summer_top.values, color="#FF6200")
axes[0].set_facecolor(BG)
axes[0].set_title("Top 10 Countries — Summer Olympics", color=GOLD_C)
axes[0].set_xlabel("Medals", color=TEXT)

# Top countries Winter
winter_top = df[df["season"]=="Winter"]["country"].value_counts().head(10).sort_values()
axes[1].barh(winter_top.index, winter_top.values, color="#00B4D8")
axes[1].set_facecolor(BG)
axes[1].set_title("Top 10 Countries — Winter Olympics", color=GOLD_C)
axes[1].set_xlabel("Medals", color=TEXT)

plt.tight_layout()
save("08_summer_vs_winter.png")

# ══════════════════════════════════════════════════════════════
# 10. CHART 9 — MEDAL POINTS LEADERBOARD
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("🏆 Medal Points Leaderboard", fontsize=15,
             fontweight="bold", color=GOLD_C)

# Country points leaderboard (Gold=3, Silver=2, Bronze=1)
country_points = df.groupby("country")["medal_points"].sum().nlargest(15).sort_values()
colors_bar = [GOLD_C if i >= len(country_points)-3 else
              SILVER_C if i >= len(country_points)-6 else "#003580"
              for i in range(len(country_points))]
axes[0].barh(country_points.index, country_points.values, color=colors_bar)
axes[0].set_facecolor(BG)
axes[0].set_title("Top 15 Countries by Medal Points\n(Gold=3, Silver=2, Bronze=1)", color=GOLD_C)
axes[0].set_xlabel("Medal Points", color=TEXT)

# Gold medal efficiency (Gold / Total)
country_gold = df[df["medal"]=="Gold"].groupby("country").size()
country_total = df.groupby("country").size()
gold_efficiency = (country_gold / country_total * 100).dropna().nlargest(15).sort_values()
axes[1].barh(gold_efficiency.index, gold_efficiency.values, color=GOLD_C)
axes[1].set_facecolor(BG)
axes[1].set_title("Top 15 Countries — Gold Medal Efficiency %", color=GOLD_C)
axes[1].set_xlabel("Gold %", color=TEXT)

plt.tight_layout()
save("09_medal_leaderboard.png")

# ══════════════════════════════════════════════════════════════
# 11. CHART 10 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG)

# Encode for correlation
df_enc = df.copy()
df_enc["medal_num"]   = df_enc["medal"].map({"Gold":3,"Silver":2,"Bronze":1})
df_enc["gender_num"]  = df_enc["gender"].map({"M":1,"F":0})
df_enc["season_num"]  = df_enc["season"].map({"Summer":1,"Winter":0})

num_cols = ["age","year","medal_num","gender_num","season_num","medal_points"]
corr = df_enc[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", ax=ax, linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={"color":"white"})
ax.set_facecolor(BG)
ax.set_title("Correlation Heatmap — Numerical Features",
             fontsize=14, fontweight="bold", color=GOLD_C)
plt.tight_layout()
save("10_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════
# 12. KEY METRICS SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("🏅 KEY METRICS SUMMARY")
print("="*55)
print(f"  Total Records       : {len(df):,}")
print(f"  Year Range          : {df['year'].min()} — {df['year'].max()}")
print(f"  Total Countries     : {df['country'].nunique()}")
print(f"  Total Sports        : {df['sport'].nunique()}")
print(f"  Total Athletes      : {df['athlete_name'].nunique():,}")
print(f"  Gold Medals         : {(df['medal']=='Gold').sum():,}")
print(f"  Silver Medals       : {(df['medal']=='Silver').sum():,}")
print(f"  Bronze Medals       : {(df['medal']=='Bronze').sum():,}")
print(f"  Avg Athlete Age     : {df['age'].mean():.1f} years")
print(f"  Youngest Medalist   : {df['age'].min()} years")
print(f"  Oldest Medalist     : {df['age'].max()} years")
print(f"  Top Country         : {df['country'].value_counts().idxmax()}")
print(f"  Top Sport           : {df['sport'].value_counts().idxmax()}")
print(f"  Summer Medals       : {(df['season']=='Summer').sum():,}")
print(f"  Winter Medals       : {(df['season']=='Winter').sum():,}")
print("="*55)
print("\n✅ All 10 charts saved to /charts folder!")
