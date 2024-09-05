import pandas as pd

def filter_by_week(input_csv, output_csv, week_id):
    df = pd.read_csv(input_csv)
    
    df_filtered = df[df['week_id'] == week_id]
    
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"Filtered data for week {week_id} saved to {output_csv}")
    print(f"Removed {len(df) - len(df_filtered)} entries")

filter_by_week('game.csv', 'week1.csv', 'nfl-2024-week-1')
