from flask import Flask, request, jsonify
import math 
from setup import setup
import argparse
import sqlite3
import pandas as pd

### GLOBALS
global sample

app = Flask(__name__)

def reset():
    setup()

def radec_to_cartesian(ra, dec, distance):
    ra_rad = ra / 12 * math.pi  # Convert RA to radians
    dec_rad = dec / 180 * math.pi  # Convert DEC to radians
    
    x = distance * math.cos(ra_rad) * math.cos(dec_rad)
    y = distance * math.sin(ra_rad) * math.cos(dec_rad)
    z = distance * math.sin(dec_rad)
    
    return {'x': x, 'y': y, 'z': z}

def cartesian_to_radec(x, y, z):
    # Compute distance (radius)
    distance = math.sqrt(x**2 + y**2 + z**2)

    ra = math.atan2(y, x)  # atan2 handles the quadrant issues
    ra = ra * 12 / math.pi  # Convert RA to degrees
    dec = math.atan2(z, math.sqrt(x**2 + y**2))  # atan2(z, r_xy)
    dec = dec * 180 / math.pi

    return {'ra': ra, 'dec': dec, 'distance': distance}

# Brightness is a size constraint
def brightness(vmag): 
    return 75 * math.pow(1.35, math.min(-vmag, 0.15))

### TODO
# To calculate relative ra dec from planet
# Subtract cartesian of planet from star, revert back to ra dec
def new_labels():
    pass

# TESTING ONLY
@app.route('/radec_to_cartesian', methods=['GET'])
def radec_to_cartesian():
    ra = float(request.args.get('ra'))
    dec = float(request.args.get('dec'))
    distance = float(request.args.get('distance'))
    result = radec_to_cartesian(ra, dec, distance)
    return jsonify(result)

@app.route('/selector')
def selector(): 
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query(f"SELECT * from planets", conn)
    df_sample = df.sample(n=1000).sort_index(ascending=True)
    ret = df_sample[['pl_name', 'pl_bmasse', 'pl_orbper', 'pl_eqt', 'pl_orbeccen', 'pl_insol', 'st_spectype', 'st_teff', 'st_rad']]
    global sample
    sample = ret
    return ret.to_json(orient="index")

@app.route('/render')
def render():
    selected_idx = int(request.args.get('index'))
    global sample
    col = sample.iloc[:, selected_idx]
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query(f"SELECT * from stars", conn)
    df_sorted = df.sort_values(by='sy_dist', ascending=True)
    df_sorted = df_sorted.head(1000)
    df_sorted = df_sorted[['sy_name', 'ra', 'dec', 'sy_dist', 'sy_vmag']]
    df_sorted['sy_vmag'] = df_sorted['sy_vmag'].apply(brightness)
    planet_json = col.to_json()
    stars_json = df_sorted.to_json(orient='index')
    return jsonify({"planet": planet_json, "stars": stars_json})

@app.route('/')
def home():
    return "Welcome to the Exosky backend"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional DB reset")
    parser.add_argument('--reset', action='store_true', help="Reset the DB")
    args = parser.parse_args()
    if args.reset:
        print("DB reset initated")
        reset()
    app.run(debug=True)
