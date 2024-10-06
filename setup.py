### ** IMPORTANT **
### DO NOT RUN! THIS RESETS THE SQL FILES
### THIS IS ONLY FOR PROJECT INITIALIZATION IN HEROKU

import sqlite3
import pandas as pd

def setup():
    csv_planets = 'data/planets.csv'
    csv_stars = 'data/stars.csv'
    csv_koi = 'data/koi.csv'

    conn = sqlite3.connect('data.db')

    planets_df = pd.read_csv(csv_planets, comment='#')
    stars_df = pd.read_csv(csv_stars, comment='#')
    koi_df = pd.read_csv(csv_koi, comment='#')

    planets_df.to_sql('planets', conn, if_exists='replace', index=False)
    stars_df.to_sql('stars', conn, if_exists='replace', index=False)
    koi_df.to_sql('koi', conn, if_exists='replace', index=False)

    conn.close()

    print("Data loaded")
