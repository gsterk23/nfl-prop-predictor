import streamlit as st
import nflreadpy as nfl
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NFL Prop Predictor",
    page_icon="🏈",
    layout="centered"
)

# ============================================================
# DATA LOADING — cached so it only runs once
# ============================================================
@st.cache_data
def load_all_data(seasons):
    df = nfl.load_player_stats(seasons).to_pandas()
    schedule = nfl.load_schedules(seasons).to_pandas()
    schedule = schedule[['game_id', 'home_team', 'away_team']].drop_duplicates()
    return df, schedule


@st.cache_data
def build_features(seasons, position):
    df, schedule = load_all_data(seasons)

    if position == 'RB':
        stat_col = 'rushing_yards'
        usage_col = 'carries'
        epa_col = 'rushing_epa'
    elif position in ['WR', 'TE']:
        stat_col = 'receiving_yards'
        usage_col = 'targets'
        epa_col = 'receiving_epa'
    elif position == 'QB':
        stat_col = 'passing_yards'
        usage_col = 'attempts'
        epa_col = 'passing_epa'

    pos_df = df[df['position'] == position].copy()
    pos_df = pos_df.sort_values(
        ['player_id', 'season', 'week']
    ).reset_index(drop=True)
    pos_df[stat_col] = pos_df[stat_col].fillna(0)
    pos_df = pos_df.merge(schedule, on='game_id', how='left')
    pos_df['is_home'] = (pos_df['team'] == pos_df['home_team']).astype(int)

    features = []
    for player_id, player_df in pos_df.groupby('player_id'):
        player_df = player_df.reset_index(drop=True)
        for i in range(len(player_df)):
            row = player_df.iloc[i]
            past_games = player_df.iloc[:i]
            if len(past_games) < 3:
                continue

            current_season_games = past_games[
                past_games['season'] == row['season']
            ]
            season_avg = current_season_games[stat_col].mean() \
                if len(current_season_games) > 0 else 0

            last_5 = past_games.tail(5)[stat_col].values
            weights = np.arange(1, len(last_5) + 1)
            l5_weighted_avg = np.average(last_5, weights=weights) \
                if len(last_5) > 0 else 0

            last_3 = past_games.tail(3)[stat_col].mean() \
                if len(past_games) >= 3 else 0
            prev_3 = past_games.iloc[-6:-3][stat_col].mean() \
                if len(past_games) >= 6 else last_3
            trend = last_3 - prev_3

            consistency = (current_season_games[stat_col] > season_avg).mean() \
                if season_avg > 0 and len(current_season_games) > 0 else 0.5

            usage = current_season_games[usage_col].mean() \
                if usage_col in current_season_games.columns else 0

            std_dev = past_games[stat_col].std() \
                if len(past_games) > 1 else 0

            is_home = int(row.get('is_home', 0))
            home_games = current_season_games[
                current_season_games['is_home'] == 1
            ]
            away_games = current_season_games[
                current_season_games['is_home'] == 0
            ]
            home_avg = home_games[stat_col].mean() \
                if len(home_games) > 0 else season_avg
            away_avg = away_games[stat_col].mean() \
                if len(away_games) > 0 else season_avg
            home_away_diff = home_avg - away_avg

            efficiency = current_season_games[epa_col].mean() \
                if epa_col in current_season_games.columns else 0

            if position in ['WR', 'TE']:
                target_share = current_season_games['target_share'].mean() \
                    if 'target_share' in current_season_games.columns else 0
                air_yards_share = current_season_games['air_yards_share'].mean() \
                    if 'air_yards_share' in current_season_games.columns else 0
                wopr = current_season_games['wopr'].mean() \
                    if 'wopr' in current_season_games.columns else 0
            else:
                target_share = 0
                air_yards_share = 0
                wopr = 0

            features.append({
                'player_id': player_id,
                'player_name': row.get('player_display_name', ''),
                'season': row['season'],
                'week': row['week'],
                'opponent_team': row.get('opponent_team', ''),
                'actual_yards': row[stat_col],
                'season_avg': round(season_avg, 2),
                'l5_weighted_avg': round(l5_weighted_avg, 2),
                'trend': round(trend, 2),
                'consistency': round(consistency, 3),
                'usage_rate': round(usage, 2),
                'std_dev': round(std_dev, 2),
                'is_home': is_home,
                'home_away_diff': round(home_away_diff, 2),
                'target_share': round(target_share, 3),
                'air_yards_share': round(air_yards_share, 3),
                'wopr': round(wopr, 3),
                'efficiency_epa': round(efficiency, 3),
            })

    return pd.DataFrame(features)


@st.cache_data
def build_def_ratings(seasons, position):
    df = nfl.load_player_stats(seasons).to_pandas()

    if position == 'RB':
        stat_col = 'rushing_yards'
        usage_col = 'carries'
    elif position in ['WR', 'TE']:
        stat_col = 'receiving_yards'
        usage_col = 'targets'
    elif position == 'QB':
        stat_col = 'passing_yards'
        usage_col = 'attempts'

    pos_df = df[df['position'] == position].copy()
    pos_df[stat_col] = pos_df[stat_col].fillna(0)
    pos_df[usage_col] = pos_df[usage_col].fillna(0)

    idx = pos_df.groupby(
        ['opponent_team', 'season', 'week']
    )[usage_col].idxmax()
    primary_df = pos_df.loc[idx]

    def_ratings = (
        primary_df.groupby(['opponent_team', 'season'])[stat_col]
        .mean()
        .reset_index()
        .rename(columns={stat_col: 'def_yards_allowed_pg'})
    )
    def_ratings['def_yards_allowed_pg'] = \
        def_ratings['def_yards_allowed_pg'].round(2)
    return def_ratings


@st.cache_resource
def train_model(seasons, position):
    feature_df = build_features(seasons, position)
    def_ratings = build_def_ratings(seasons, position)

    feature_df = feature_df.merge(
        def_ratings, on=['opponent_team', 'season'], how='left'
    )
    league_avg = def_ratings['def_yards_allowed_pg'].mean()
    feature_df['def_yards_allowed_pg'] = \
        feature_df['def_yards_allowed_pg'].fillna(league_avg)

    if position == 'RB':
        feature_cols = [
            'season_avg', 'l5_weighted_avg', 'trend',
            'consistency', 'usage_rate', 'std_dev',
            'def_yards_allowed_pg',
        ]
    elif position in ['WR', 'TE']:
        feature_cols = [
            'season_avg', 'l5_weighted_avg', 'trend',
            'consistency', 'usage_rate', 'std_dev',
            'def_yards_allowed_pg', 'target_share',
            'air_yards_share', 'wopr', 'efficiency_epa',
            'home_away_diff',
        ]
    elif position == 'QB':
        feature_cols = [
            'season_avg', 'l5_weighted_avg', 'trend',
            'consistency', 'usage_rate', 'std_dev',
            'def_yards_allowed_pg', 'efficiency_epa',
            'home_away_diff',
        ]

    model_df = feature_df[feature_cols + ['actual_yards']].dropna()
    X = model_df[feature_cols]
    y = model_df['actual_yards']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, feature_cols, feature_df, def_ratings, mae, r2


# ============================================================
# CHART
# ============================================================
def plot_distribution(player_name, predicted_yards, std_dev, prop_line, position):
    if position == 'RB':
        yard_label = "Rushing Yards"
    elif position in ['WR', 'TE']:
        yard_label = "Receiving Yards"
    else:
        yard_label = "Passing Yards"

    x = np.linspace(
        predicted_yards - 4 * std_dev,
        predicted_yards + 4 * std_dev, 300
    )
    y = stats.norm.pdf(x, loc=predicted_yards, scale=std_dev)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, color='royalblue', linewidth=2.5,
            label='Predicted Distribution')

    x_over = x[x >= prop_line]
    y_over = stats.norm.pdf(x_over, loc=predicted_yards, scale=std_dev)
    ax.fill_between(x_over, y_over, alpha=0.4,
                    color='green', label='Over Region')

    x_under = x[x <= prop_line]
    y_under = stats.norm.pdf(x_under, loc=predicted_yards, scale=std_dev)
    ax.fill_between(x_under, y_under, alpha=0.4,
                    color='red', label='Under Region')

    ax.axvline(x=prop_line, color='orange', linewidth=2,
               linestyle='--', label=f'Prop Line: {prop_line} yds')
    ax.axvline(x=predicted_yards, color='royalblue', linewidth=2,
               linestyle='-', label=f'Predicted: {predicted_yards:.1f} yds')

    prob_over = (1 - stats.norm.cdf(
        prop_line, loc=predicted_yards, scale=std_dev)) * 100
    prob_under = 100 - prob_over

    ax.text(prop_line - std_dev, max(y) * 0.5,
            f'UNDER\n{prob_under:.1f}%',
            fontsize=13, color='red', fontweight='bold', ha='center')
    ax.text(prop_line + std_dev, max(y) * 0.5,
            f'OVER\n{prob_over:.1f}%',
            fontsize=13, color='green', fontweight='bold', ha='center')

    ax.set_title(
        f'{player_name} — {yard_label} Probability Distribution',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel(yard_label, fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================
# UI
# ============================================================
st.title("🏈 NFL Prop Predictor")
st.markdown("*Powered by XGBoost + Normal Distribution*")
st.divider()

seasons = [2022, 2023, 2024, 2025]

# --- Sidebar inputs ---
with st.sidebar:
    st.header("⚙️ Settings")
    position = st.selectbox(
        "Position",
        options=['RB', 'WR', 'TE', 'QB'],
        index=0
    )

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Select a position")
    st.markdown("2. Enter a player name")
    st.markdown("3. Enter the opponent + prop line")
    st.markdown("4. Get a probability instantly")

# --- Load and train ---
with st.spinner(f"Loading data and training {position} model... (30-60 sec)"):
    model, feature_cols, feature_df, def_ratings, mae, r2 = train_model(
        tuple(seasons), position
    )

st.success(f"✅ {position} model ready — MAE: {mae:.1f} yds | R²: {r2:.3f}")
st.divider()

# --- Player input ---
col1, col2 = st.columns(2)

with col1:
    player_name = st.text_input("Player Name", placeholder="e.g. Derrick Henry")

with col2:
    all_teams = sorted(def_ratings['opponent_team'].unique().tolist())
    opponent = st.selectbox("Opponent Team", options=all_teams)

col3, col4 = st.columns(2)

with col3:
    prop_line = st.number_input(
        "Prop Line (yards)", min_value=0.0,
        max_value=600.0, value=86.5, step=0.5
    )

with col4:
    home_away = st.radio(
        "Home or Away?",
        options=["Home", "Away"],
        horizontal=True
    )
    is_home = 1 if home_away == "Home" else 0

# --- Predict button ---
if st.button("🔮 Predict", use_container_width=True, type="primary"):
    if not player_name:
        st.warning("Please enter a player name.")
    else:
        player_df = feature_df[
            feature_df['player_name'].str.lower() == player_name.lower()
        ].copy()

        if player_df.empty:
            # Try partial match
            partial = feature_df[
                feature_df['player_name'].str.lower().str.contains(
                    player_name.lower()
                )
            ]['player_name'].unique()

            if len(partial) > 0:
                st.error(f"Player not found. Did you mean:")
                for name in partial[:5]:
                    st.markdown(f"- **{name}**")
            else:
                st.error(
                    f"No player found matching '{player_name}'. "
                    f"Check spelling and ensure they played 2022-2025."
                )
        else:
            # Apply opponent defensive rating
            opponent_def = def_ratings[
                def_ratings['opponent_team'] == opponent
            ].sort_values('season', ascending=False).iloc[0]
            opponent_def_rating = opponent_def['def_yards_allowed_pg']

            player_df['def_yards_allowed_pg'] = opponent_def_rating
            player_df['is_home'] = is_home

            # Merge def ratings for feature df
            feature_df_merged = feature_df.merge(
                def_ratings, on=['opponent_team', 'season'], how='left'
            )

            latest = player_df.dropna(subset=feature_cols).iloc[-1]
            predicted_yards = model.predict([latest[feature_cols]])[0]
            historical_std = player_df['std_dev'].iloc[-1]

            prob_over = 1 - stats.norm.cdf(
                prop_line, loc=predicted_yards, scale=historical_std
            )
            prob_under = 1 - prob_over

            if prob_over >= 0.65:
                signal = "🟢 STRONG OVER"
            elif prob_over >= 0.55:
                signal = "🟡 LEAN OVER"
            elif prob_over <= 0.35:
                signal = "🔴 STRONG UNDER"
            elif prob_over <= 0.45:
                signal = "🟠 LEAN UNDER"
            else:
                signal = "⚪ TOSS UP"

            # --- Results display ---
            st.divider()
            st.subheader(f"📊 {player_name} vs {opponent}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Yards", f"{predicted_yards:.1f}")
            m2.metric("Prop Line", f"{prop_line}")
            m3.metric("Std Dev", f"{historical_std:.1f}")

            st.divider()

            c1, c2, c3 = st.columns(3)
            c1.metric("OVER %", f"{prob_over*100:.1f}%")
            c2.metric("UNDER %", f"{prob_under*100:.1f}%")
            c3.metric("Signal", signal)

            st.divider()

            # --- Distribution chart ---
            fig = plot_distribution(
                player_name, predicted_yards,
                historical_std, prop_line, position
            )
            st.pyplot(fig)

            # --- Player stats context ---
            with st.expander("📋 View Player Feature Details"):
                display_cols = [
                    'season', 'week', 'season_avg', 'l5_weighted_avg',
                    'trend', 'consistency', 'usage_rate', 'std_dev'
                ]
                available = [c for c in display_cols if c in player_df.columns]
                st.dataframe(player_df[available].tail(10))

            # --- Defensive context ---
            st.info(
                f"📌 {opponent} allowed **{opponent_def_rating:.1f} yds/game** "
                f"to opposing {position}s in {int(opponent_def['season'])}"
            )
