import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df['spread'] = df['spread'].apply(lambda x: float(x.split()[1]))
    df['home_team_favored'] = df['spread'] < 0
    df['spread_magnitude'] = abs(df['spread'])
    return df

def estimate_win_probabilities(df):
    k = 0.15  # steepness of the curve
    probabilities = 1 / (1 + np.exp(-k * df['spread']))
    return probabilities

def optimize_picks(probabilities, num_games):
    # Calculate expected value and standard deviation for each game
    expected_values = probabilities * np.arange(num_games, 0, -1)
    std_devs = np.sqrt(probabilities * (1 - probabilities) * np.arange(num_games, 0, -1)**2)
    
    # Calculate the probability of each pick being in the top 10% of scores
    top_10_probs = 1 - norm.cdf(np.percentile(expected_values, 90), expected_values, std_devs)
    
    # Create cost matrix based on top 10% probabilities
    cost_matrix = -np.outer(top_10_probs, range(1, num_games + 1))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimized_picks = list(zip(row_ind, col_ind + 1))
    optimized_picks.sort(key=lambda x: x[1], reverse=True)
    return optimized_picks

def simulate_week(probabilities, picks, num_simulations=10000, num_players=100):
    num_games = len(probabilities)
    all_scores = []
    optimized_scores = []
    
    for _ in range(num_simulations):
        outcomes = np.random.random(num_games) < probabilities
        
        # Simulate other players
        for _ in range(num_players):
            auto_picks = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
            auto_picks = [(game, num_games - i) for i, (game, _) in enumerate(auto_picks)]
            
            # Introduce random changes
            if np.random.random() < 0.8:  # 80% chance of making changes
                num_changes = np.random.randint(1, 5)  # Make 1-4 changes
                for _ in range(num_changes):
                    i, j = np.random.choice(len(auto_picks), 2, replace=False)
                    auto_picks[i], auto_picks[j] = auto_picks[j], auto_picks[i]
            
            score = sum(conf for game, conf in auto_picks if outcomes[game])
            all_scores.append(score)
        
        # Calculate optimized score
        optimized_score = sum(conf for game, conf in picks if outcomes[game])
        optimized_scores.append(optimized_score)
    
    return all_scores, optimized_scores

def evaluate_strategy(optimized_picks, probabilities):
    all_scores, optimized_scores = simulate_week(probabilities, optimized_picks)
    
    print(f"Average player score: {np.mean(all_scores):.2f}")
    print(f"Average optimized score: {np.mean(optimized_scores):.2f}")
    print(f"Probability of beating average player: {np.mean(np.array(optimized_scores) > np.mean(all_scores)):.2f}")
    print(f"Probability of top 3 finish: {np.mean(np.array(optimized_scores) >= np.percentile(all_scores, 97)):.2f}")

def main():
    df = preprocess_data('week1.csv')
    probabilities = estimate_win_probabilities(df)
    optimized_picks = optimize_picks(probabilities, len(df))
    evaluate_strategy(optimized_picks, probabilities)
    
    print("\nOptimized picks:")
    for game, confidence in optimized_picks:
        favorite = "Home" if df.iloc[game]['home_team_favored'] else "Away"
        print(f"Game {game + 1}: {favorite} team, Confidence {confidence}")

if __name__ == "__main__":
    main()
