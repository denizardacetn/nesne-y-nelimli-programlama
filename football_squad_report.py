
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. AYARLAR
# --------------------------------------------------
N = 5000
SEED = 42
rng = np.random.default_rng(SEED)

OUT = Path("football_squad_report")
OUT.mkdir(exist_ok=True)

teams = [
    "North London FC", "Madrid Royal", "Catalonia FC",
    "Munich United", "Milan Red", "Turin Black",
    "Paris Athletic", "Manchester Blue", "Liverpool Red",
    "London Blue", "Istanbul Eagles", "Istanbul Lions",
    "Amsterdam Stars", "Lisbon United", "Porto Dragons",
    "Dortmund Yellow", "Rome Wolves", "Naples Blue",
    "Seville FC", "London Gunners"
]

positions = ["GK", "CB", "LB", "RB", "CDM",
             "CM", "CAM", "LW", "RW", "ST"]

position_weights = [0.09, 0.15, 0.08, 0.08, 0.10,
                    0.14, 0.09, 0.09, 0.08, 0.10]

countries = [
    "Türkiye", "Brazil", "Argentina", "France", "Spain",
    "England", "Germany", "Italy", "Portugal", "Netherlands",
    "Nigeria", "Japan", "Morocco", "Uruguay", "Croatia",
    "Belgium", "Senegal", "United States", "Sweden", "Ghana"
]

first_names = [
    "Alex", "Lucas", "Emir", "Mateo", "Noah", "Leo",
    "Arda", "Daniel", "Gabriel", "Kerem", "David",
    "Marco", "Adam", "Rafael", "Omar", "Milan",
    "Ethan", "Can", "Felix", "Hugo"
]

last_names = [
    "Silva", "Yilmaz", "Garcia", "Demir", "Rossi",
    "Martin", "Kaya", "Santos", "Brown", "Muller",
    "Costa", "Lopez", "Johnson", "Kovacic", "Moreira",
    "Dubois", "Andersson", "Fernandez", "Aydin", "Walker"
]

# --------------------------------------------------
# 2. FUTBOLCU VERİ SETİ
# --------------------------------------------------
player_positions = rng.choice(
    positions, size=N, p=position_weights
)

ages = rng.integers(17, 41, size=N)

overall = np.clip(
    np.rint(rng.normal(67, 10, N)), 35, 95
).astype(int)

potential = np.maximum(
    overall,
    np.clip(
        overall + rng.normal(7, 7, N), 35, 99
    ).astype(int)
)

appearances = rng.integers(0, 39, size=N)
minutes = np.array([
    rng.integers(0, int(app * 90) + 1)
    for app in appearances
])

is_attacker = np.isin(
    player_positions, ["ST", "LW", "RW", "CAM"]
)
is_midfielder = np.isin(
    player_positions, ["CDM", "CM", "CAM"]
)
is_defender = np.isin(
    player_positions, ["CB", "LB", "RB"]
)
is_goalkeeper = player_positions == "GK"

goal_rate = np.select(
    [is_goalkeeper, is_defender, is_midfielder, is_attacker],
    [0.001, 0.025, 0.09, 0.22],
    default=0.05
)

assist_rate = np.select(
    [is_goalkeeper, is_defender, is_midfielder, is_attacker],
    [0.005, 0.025, 0.12, 0.10],
    default=0.05
)

goals = rng.poisson(appearances * goal_rate)
assists = rng.poisson(appearances * assist_rate)

# Kaleci kurtarışları ve saha oyuncusu müdahaleleri
saves = np.where(
    is_goalkeeper,
    rng.poisson(appearances * 2.8),
    rng.poisson(appearances * 0.15)
)

tackles = np.where(
    is_goalkeeper,
    rng.poisson(appearances * 0.1),
    rng.poisson(appearances * np.where(
        is_defender, 1.8, 1.1
    ))
)

# Piyasa değeri: yaklaşık milyon avro
age_factor = np.exp(-0.025 * np.abs(ages - 25))
market_value = (
    (overall - 30) ** 2
    * age_factor
    * rng.uniform(0.015, 0.055, N)
)
market_value = np.round(
    np.clip(market_value, 0.02, 200), 2
)

weekly_wage = np.round(
    overall ** 3 * rng.uniform(0.02, 0.12, N)
).astype(int)

data = pd.DataFrame({
    "player_id": np.arange(1, N + 1),
    "player_name": [
        f"{rng.choice(first_names)} {rng.choice(last_names)}"
        for _ in range(N)
    ],
    "age": ages,
    "nationality": rng.choice(countries, N),
    "team": rng.choice(teams, N),
    "position": player_positions,
    "shirt_number": rng.integers(1, 100, N),
    "overall_rating": overall,
    "potential": potential,
    "appearances": appearances,
    "minutes_played": minutes,
    "goals": goals,
    "assists": assists,
    "saves": saves,
    "tackles": tackles,
    "market_value_m_eur": market_value,
    "weekly_wage_eur": weekly_wage,
    "contract_years": rng.integers(0, 6, N)
})

# Tekrarlanan oyuncu adlarını önlemek için
data["player_name"] = [
    f"{first_names[(i - 1) % len(first_names)]} "
    f"{last_names[((i - 1) // len(first_names)) % len(last_names)]}"
    for i in range(1, N + 1)
]

data["goals_per_90"] = np.round(
    np.divide(
        data["goals"] * 90,
        data["minutes_played"],
        out=np.zeros(N, dtype=float),
        where=data["minutes_played"].to_numpy() > 0
    ), 3
)

data["goal_contributions"] = data["goals"] + data["assists"]

# --------------------------------------------------
# 3. ÖZET TABLOLARI
# --------------------------------------------------
team_summary = data.groupby("team").agg(
    player_count=("player_id", "count"),
    avg_age=("age", "mean"),
    avg_rating=("overall_rating", "mean"),
    total_goals=("goals", "sum"),
    total_assists=("assists", "sum"),
    total_market_value=("market_value_m_eur", "sum"),
    avg_potential=("potential", "mean")
).reset_index()

position_summary = data.groupby("position").agg(
    player_count=("player_id", "count"),
    avg_age=("age", "mean"),
    avg_rating=("overall_rating", "mean"),
    avg_potential=("potential", "mean"),
    total_goals=("goals", "sum"),
    total_assists=("assists", "sum"),
    avg_market_value=("market_value_m_eur", "mean")
).reset_index()

top_players = data.sort_values(
    ["overall_rating", "potential"],
    ascending=False
).head(50)

top_values = data.nlargest(50, "market_value_m_eur")

# --------------------------------------------------
# 4. CSV VE EXCEL ÇIKTILARI
# --------------------------------------------------
data.to_csv(
    OUT / "football_players_5000.csv",
    index=False,
    encoding="utf-8-sig"
)

with pd.ExcelWriter(
    OUT / "football_squad_report.xlsx",
    engine="openpyxl"
) as writer:
    data.to_excel(
        writer, sheet_name="All Players", index=False
    )
    team_summary.round(2).to_excel(
        writer, sheet_name="Team Summary", index=False
    )
    position_summary.round(2).to_excel(
        writer, sheet_name="Position Summary", index=False
    )
    top_players.to_excel(
        writer, sheet_name="Top 50 Players", index=False
    )
    top_values.to_excel(
        writer, sheet_name="Top 50 Values", index=False
    )

# --------------------------------------------------
# 5. GRAFİKLER
# --------------------------------------------------
plt.figure(figsize=(10, 6))
team_summary.sort_values(
    "total_goals", ascending=False
).plot(
    x="team", y="total_goals", kind="bar",
    legend=False, color="seagreen", figsize=(12, 6)
)
plt.title("Total Goals by Team")
plt.xlabel("Team")
plt.ylabel("Goals")
plt.xticks(rotation=70, ha="right")
plt.tight_layout()
plt.savefig(OUT / "goals_by_team.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
data.groupby("position")["overall_rating"].mean().reindex(
    positions
).plot(kind="bar", color="steelblue")
plt.title("Average Overall Rating by Position")
plt.xlabel("Position")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.savefig(OUT / "rating_by_position.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 5))
plt.scatter(
    data["age"], data["overall_rating"],
    alpha=0.35, s=15, color="darkorange"
)
plt.title("Age vs Overall Rating")
plt.xlabel("Age")
plt.ylabel("Overall Rating")
plt.tight_layout()
plt.savefig(OUT / "age_vs_rating.png", dpi=150)
plt.close()

# --------------------------------------------------
# 6. RAPOR ÖZETİ
# --------------------------------------------------
report = f"""
FOOTBALL SQUAD REPORT
=====================

Total player records: {len(data):,}
Number of teams: {data['team'].nunique()}
Average player age: {data['age'].mean():.2f}
Average overall rating: {data['overall_rating'].mean():.2f}
Average potential: {data['potential'].mean():.2f}
Total goals: {data['goals'].sum():,}
Total assists: {data['assists'].sum():,}
Total market value: {data['market_value_m_eur'].sum():,.2f} million EUR

Highest-rated player:
{data.loc[data['overall_rating'].idxmax(),
          ['player_name', 'team', 'position',
           'overall_rating']].to_string()}

Top team by goals:
{team_summary.loc[team_summary['total_goals'].idxmax(),
                  ['team', 'total_goals']].to_string()}

NOTE:
This dataset is synthetic and intended for practice.
It does not represent verified real-world football data.
"""

(OUT / "report_summary.txt").write_text(
    report, encoding="utf-8"
)

print(report)
print(f"\nAll outputs saved to: {OUT.resolve()}")
