from supabase import create_client
import pandas as pd
import os

url = "https://zwgibswpucbhxmlpouhw.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp3Z2lic3dwdWNiaHhtbHBvdWh3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyMTkwNTMsImV4cCI6MjAzNzc5NTA1M30.MryGEXFq7O5lnAbD0ldxDlB_fSH3wSMoov0jlZFZJjg"
supabase = create_client(url, key)

def fetch_and_save_as_csv(table_name):
    response = supabase.table(table_name).select("*").execute()
    
    df = pd.DataFrame(response.data)
    
    csv_filename = f"{table_name}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved {csv_filename}")

fetch_and_save_as_csv("game")
