from supabase import create_client
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

def fetch_and_save_as_csv(table_name):
    response = supabase.table(table_name).select("*").execute()
    
    df = pd.DataFrame(response.data)
    
    csv_filename = f"{table_name}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved {csv_filename}")

fetch_and_save_as_csv("game")
