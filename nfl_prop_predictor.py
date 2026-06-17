import nflreadpy as nfl
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

from scipy import stats

def load_player_game_logs(seasons):
    print(f"Loading player game logs for seasons: {seasons}...")
    df = nfl.load_player_stats(seasons)
    df = df.to_pandas()
    print(f"Loaded {len(df)} rows of data.")
    return df


def load_schedule(seasons):
    """
    Pulls NFL schedule data to get home/away info for each game.
    """
    print("Loading schedule data for home/away splits...")
    schedule = nfl.load_schedules(seasons).to_pandas()

    # We need game_id, home_team, away_team
    schedule = schedule[['game_id', 'home_team', 'away_team']].drop_duplicates()
    return schedule


def filter_by_position(df, position):
    filtered = df[df['position'] == position].copy()
    print(f"Filtered to {position}: {len(filtered)} rows")
    return filtered

def engineer_features(df, position, schedule):
    """
    Builds predictive features for a given position group.
    Now includes home/away splits, enhanced usage metrics,
    and efficiency ratings.
    """
    print(f"\nEngineering features for {position}s...")

    # Set the target stat column based on position
    if position == 'RB':
        stat_col = 'rushing_yards'
    elif position in ['WR', 'TE']:
        stat_col = 'receiving_yards'
    elif position == 'QB':
        stat_col = 'passing_yards'

    # Filter to position and sort
    pos_df = df[df['position'] == position].copy()
    pos_df = pos_df.sort_values(
        ['player_id', 'season', 'week']
    ).reset_index(drop=True)

    # Fill missing values
    pos_df[stat_col] = pos_df[stat_col].fillna(0)

    # --- Merge home/away from schedule ---
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

            # --- Feature 1: Last 5 weighted average ---
            last_5 = past_games.tail(5)[stat_col].values
            if len(last_5) > 0:
                weights = np.arange(1, len(last_5) + 1)
                l5_weighted_avg = np.average(last_5, weights=weights)
            else:
                l5_weighted_avg = 0

            # --- Feature 2: Trend ---
            last_3 = past_games.tail(3)[stat_col].mean() \
                if len(past_games) >= 3 else 0
            prev_3 = past_games.iloc[-6:-3][stat_col].mean() \
                if len(past_games) >= 6 else last_3
            trend = last_3 - prev_3

            # --- Feature 3: Consistency ---
            if season_avg > 0 and len(current_season_games) > 0:
                consistency = (
                    current_season_games[stat_col] > season_avg
                ).mean()
            else:
                consistency = 0.5

            # --- Feature 4: Usage rate ---
            if position == 'RB':
                usage_col = 'carries'
            elif position in ['WR', 'TE']:
                usage_col = 'targets'
            elif position == 'QB':
                usage_col = 'attempts'

            usage = current_season_games[usage_col].mean() \
                if usage_col in current_season_games.columns else 0

            # --- Feature 5: Standard deviation ---
            std_dev = past_games[stat_col].std() \
                if len(past_games) > 1 else 0

            # --- Feature 6: Home/Away split ---
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

            # --- Feature 7: Enhanced usage metrics ---
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

            # --- Feature 8: Efficiency rating ---
            if position == 'RB':
                epa_col = 'rushing_epa'
            elif position in ['WR', 'TE']:
                epa_col = 'receiving_epa'
            elif position == 'QB':
                epa_col = 'passing_epa'

            efficiency = current_season_games[epa_col].mean() \
                if epa_col in current_season_games.columns else 0

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

    feature_df = pd.DataFrame(features)
    print(f"Built {len(feature_df)} player-game feature rows.")
    return feature_df


def preview_player_features(feature_df, player_name):
    player_df = feature_df[feature_df['player_name'].str.lower() == player_name.lower()]
    if player_df.empty:
        print(f"Player '{player_name}' not found in feature set.")
        return
    print(f"\nFeature preview for {player_name} (last 10 games):")
    print(player_df.tail(10).to_string(index=False))

def load_defensive_ratings(seasons, position):
    """
    Calculates how many yards each NFL defense allows per game
    to the PRIMARY ball carrier only (not spread across all backs).
    This gives a more accurate picture of starter-level exposure.
    """
    print(f"\nLoading defensive ratings for seasons: {seasons}...")
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

    # For each game, only keep the player with the most carries/targets/attempts
    # This isolates the primary ball carrier against each defense
    idx = pos_df.groupby(['opponent_team', 'season', 'week'])[usage_col].idxmax()
    primary_carrier_df = pos_df.loc[idx]

    # Now average yards allowed to that primary carrier per game
    def_ratings = (
        primary_carrier_df.groupby(['opponent_team', 'season'])[stat_col]
        .mean()
        .reset_index()
        .rename(columns={stat_col: 'def_yards_allowed_pg'})
    )

    def_ratings['def_yards_allowed_pg'] = def_ratings['def_yards_allowed_pg'].round(2)

    print(f"Defensive ratings calculated for {def_ratings['opponent_team'].nunique()} teams.")
    print("\nTop 5 most generous defenses:")
    print(def_ratings.sort_values('def_yards_allowed_pg', ascending=False).head(5).to_string(index=False))
    print("\nTop 5 toughest defenses:")
    print(def_ratings.sort_values('def_yards_allowed_pg', ascending=True).head(5).to_string(index=False))
    return def_ratings


def merge_defensive_ratings(feature_df, def_ratings):
    """
    Merges defensive ratings into our feature dataframe.
    Each player-game row gets the opponent's defensive rating attached.
    """
    merged = feature_df.merge(
        def_ratings,
        on=['opponent_team', 'season'],
        how='left'
    )

    # Fill any missing defensive ratings with the league average
    league_avg = def_ratings['def_yards_allowed_pg'].mean()
    merged['def_yards_allowed_pg'] = merged['def_yards_allowed_pg'].fillna(league_avg)

    print(f"\nMerged defensive ratings. Shape: {merged.shape}")
    return merged


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

def train_model(feature_df, position):
    print("\nTraining prediction model...")

    # Position specific feature sets
    # RBs are volume/usage driven — keep it lean
    if position == 'RB':
        feature_cols = [
            'season_avg',
            'l5_weighted_avg',
            'trend',
            'consistency',
            'usage_rate',
            'std_dev',
            'def_yards_allowed_pg',
        ]
    # WRs/TEs benefit from target share and air yards metrics
    elif position in ['WR', 'TE']:
        feature_cols = [
            'season_avg',
            'l5_weighted_avg',
            'trend',
            'consistency',
            'usage_rate',
            'std_dev',
            'def_yards_allowed_pg',
            'target_share',
            'air_yards_share',
            'wopr',
            'efficiency_epa',
            'home_away_diff',
        ]
    # QBs are heavily game script dependent
    elif position == 'QB':
        feature_cols = [
            'season_avg',
            'l5_weighted_avg',
            'trend',
            'consistency',
            'usage_rate',
            'std_dev',
            'def_yards_allowed_pg',
            'efficiency_epa',
            'home_away_diff',
        ]

    target_col = 'actual_yards'

    model_df = feature_df[feature_cols + [target_col]].dropna()

    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance on Test Set:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} yards")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")

    print(f"\nFeature Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.to_string(index=False))

    return model, feature_cols

def preview_predictions(model, feature_df, feature_cols, player_name):
    """
    Shows predicted vs actual yards for a specific player.
    """
    player_df = feature_df[
        feature_df['player_name'].str.lower() == player_name.lower()
    ].copy()

    if player_df.empty:
        print(f"Player '{player_name}' not found.")
        return

    player_df = player_df.dropna(subset=feature_cols)
    player_df['predicted_yards'] = model.predict(player_df[feature_cols]).round(1)
    player_df['difference'] = (player_df['predicted_yards'] - player_df['actual_yards']).round(1)

    print(f"\nPredicted vs Actual for {player_name} (last 10 games):")
    cols = ['season', 'week', 'opponent_team', 'actual_yards', 'predicted_yards', 'difference']
    print(player_df[cols].tail(10).to_string(index=False))


from scipy import stats


def calculate_probability(model, feature_df, feature_cols, player_name, prop_line, position):

    # Set yard label based on position
    if position == 'RB':
        yard_label = "Rushing Yards"
    elif position in ['WR', 'TE']:
        yard_label = "Receiving Yards"
    elif position == 'QB':
        yard_label = "Passing Yards"

    player_df = feature_df[
        feature_df['player_name'].str.lower() == player_name.lower()
    ].dropna(subset=feature_cols)

    if player_df.empty:
        print(f"Player '{player_name}' not found.")
        return

    latest_features = player_df[feature_cols].iloc[-1]
    predicted_yards = model.predict([latest_features])[0]
    historical_std = player_df['std_dev'].iloc[-1]

    prob_over = 1 - stats.norm.cdf(prop_line, loc=predicted_yards, scale=historical_std)
    prob_under = 1 - prob_over

    print(f"\n{'='*50}")
    print(f"  PROP PREDICTION: {player_name}")
    print(f"{'='*50}")
    print(f"  Prop Line:         {prop_line} {yard_label}")
    print(f"  Predicted {yard_label}:   {predicted_yards:.1f}")
    print(f"  Historical StdDev: {historical_std:.1f} yards")
    print(f"{'='*50}")
    print(f"  OVER probability:  {prob_over*100:.1f}%")
    print(f"  UNDER probability: {prob_under*100:.1f}%")
    print(f"{'='*50}")

    if prob_over >= 0.65:
        signal = "STRONG OVER"
    elif prob_over >= 0.55:
        signal = "LEAN OVER"
    elif prob_over <= 0.35:
        signal = "STRONG UNDER"
    elif prob_over <= 0.45:
        signal = "LEAN UNDER"
    else:
        signal = "TOSS UP"

    print(f"  Signal:            {signal}")
    print(f"{'='*50}\n")

    return prob_over, prob_under, predicted_yards


def visualize_distribution(player_name, predicted_yards, historical_std, prop_line, position):

    # Set yard label based on position
    if position == 'RB':
        yard_label = "Rushing Yards"
    elif position in ['WR', 'TE']:
        yard_label = "Receiving Yards"
    elif position == 'QB':
        yard_label = "Passing Yards"

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(predicted_yards - 4*historical_std,
                    predicted_yards + 4*historical_std, 300)
    y = stats.norm.pdf(x, loc=predicted_yards, scale=historical_std)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, y, color='royalblue', linewidth=2.5, label='Predicted Distribution')

    x_over = x[x >= prop_line]
    y_over = stats.norm.pdf(x_over, loc=predicted_yards, scale=historical_std)
    ax.fill_between(x_over, y_over, alpha=0.4, color='green', label='Over Region')

    x_under = x[x <= prop_line]
    y_under = stats.norm.pdf(x_under, loc=predicted_yards, scale=historical_std)
    ax.fill_between(x_under, y_under, alpha=0.4, color='red', label='Under Region')

    ax.axvline(x=prop_line, color='orange', linewidth=2,
               linestyle='--', label=f'Prop Line: {prop_line} yds')
    ax.axvline(x=predicted_yards, color='royalblue', linewidth=2,
               linestyle='-', label=f'Predicted: {predicted_yards:.1f} yds')

    prob_over = (1 - stats.norm.cdf(prop_line, loc=predicted_yards,
                                     scale=historical_std)) * 100
    prob_under = 100 - prob_over

    ax.text(prop_line - historical_std, max(y)*0.5,
            f'UNDER\n{prob_under:.1f}%',
            fontsize=13, color='red', fontweight='bold', ha='center')
    ax.text(prop_line + historical_std, max(y)*0.5,
            f'OVER\n{prob_over:.1f}%',
            fontsize=13, color='green', fontweight='bold', ha='center')

    ax.set_title(f'{player_name} — {yard_label} Probability Distribution',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(yard_label, fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prop_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved as 'prop_distribution.png'")


if __name__ == "__main__":
    print("=" * 50)
    print("   NFL PROP PREDICTOR")
    print("   Powered by XGBoost + Normal Distribution")
    print("=" * 50)

    seasons = [2022, 2023, 2024, 2025]

    # Load schedule once — used for home/away splits
    schedule = load_schedule(seasons)

    while True:
        print("\nAvailable positions: RB, WR, TE, QB")
        position = input("Enter position: ").strip().upper()

        if position not in ['RB', 'WR', 'TE', 'QB']:
            print("Invalid position. Please enter RB, WR, TE, or QB.")
            continue

        print(f"\nLoading and training model for {position}s...")
        print("This may take 30-60 seconds...")

        df = load_player_game_logs(seasons)
        feature_df = engineer_features(df, position, schedule)
        def_ratings = load_defensive_ratings(seasons, position)
        feature_df = merge_defensive_ratings(feature_df, def_ratings)
        model, feature_cols = train_model(feature_df, position)

        print("\nModel ready! You can now look up any player.")

        while True:
            print("\n" + "-" * 50)
            player_name = input(
                "Enter player name (or 'quit' to exit): "
            ).strip()

            if player_name.lower() == 'quit':
                print("\nThanks for using NFL Prop Predictor. Good luck!")
                exit()

            matches = feature_df[
                feature_df['player_name'].str.lower() == player_name.lower()
            ]

            if matches.empty:
                partial = feature_df[
                    feature_df['player_name'].str.lower().str.contains(
                        player_name.lower()
                    )
                ]['player_name'].unique()

                if len(partial) > 0:
                    print(f"\nPlayer not found. Did you mean one of these?")
                    for name in partial[:5]:
                        print(f"  - {name}")
                else:
                    print(f"\nNo player found matching '{player_name}'.")
                    print(
                        "Make sure spelling is correct and "
                        "they played 2022-2025."
                    )
                continue

            print("\nNFL Team Abbreviations:")
            print("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN")
            print("DET GB  HOU IND JAX KC  LA  LAC LV  MIA")
            print("MIN NE  NO  NYG NYJ PHI PIT SEA SF  TB")
            print("TEN WAS")
            opponent = input(
                f"Enter opponent team abbreviation: "
            ).strip().upper()

            valid_opponents = def_ratings['opponent_team'].unique()
            if opponent not in valid_opponents:
                print(f"Team '{opponent}' not found.")
                print(f"Valid teams: {sorted(valid_opponents)}")
                continue

            # Ask home or away
            home_away = input(
                f"Is {player_name} playing at HOME or AWAY? (home/away): "
            ).strip().lower()
            is_home = 1 if home_away == 'home' else 0

            try:
                prop_line = float(
                    input(f"Enter prop line for {player_name}: ").strip()
                )
            except ValueError:
                print("Invalid prop line. Please enter a number like 86.5")
                continue

            # Get opponent defensive rating
            opponent_def = def_ratings[
                def_ratings['opponent_team'] == opponent
            ].sort_values('season', ascending=False).iloc[0]
            opponent_def_rating = opponent_def['def_yards_allowed_pg']
            opponent_season = opponent_def['season']

            print(
                f"\n  {opponent} allowed {opponent_def_rating:.1f} yds/game "
                f"to opposing {position}s in {int(opponent_season)}"
            )

            # Override defensive rating and home/away in feature row
            player_df = feature_df[
                feature_df['player_name'].str.lower() == player_name.lower()
            ].copy()
            player_df['def_yards_allowed_pg'] = opponent_def_rating
            player_df['is_home'] = is_home

            result = calculate_probability(
                model, player_df, feature_cols, player_name, prop_line, position
            )

            if result is None:
                continue

            prob_over, prob_under, predicted_yards = result

            show_chart = input(
                "Show distribution chart? (yes/no): "
            ).strip().lower()
            if show_chart in ['yes', 'y']:
                player_std = feature_df[
                    feature_df['player_name'].str.lower() == player_name.lower()
                ]['std_dev'].iloc[-1]

                visualize_distribution(
                    player_name, predicted_yards,
                    player_std, prop_line, position
                )

            print("\nWhat would you like to do next?")
            print("  1 - Look up another player (same position)")
            print("  2 - Switch to a different position")
            print("  3 - Quit")
            choice = input("Enter choice (1/2/3): ").strip()

            if choice == '1':
                continue
            elif choice == '2':
                break
            elif choice == '3':
                print("\nThanks for using NFL Prop Predictor. Good luck!")
                exit()
            else:
                print("Invalid choice, returning to player lookup.")
                continue
