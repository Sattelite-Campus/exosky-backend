from flask import Flask, request, jsonify
import math 
from setup import setup
import argparse
import sqlite3
import pandas as pd
#from openai import OpenAI

### GLOBALS
global sample

### CONSTANTS
TEXT_BOT = """You are telling an awe-inspiring story about the stars in the night sky."""
IMAGE_BOT = """Tell an awe-inspiring story about this constellation!"""

app = Flask(__name__)
#client = OpenAI()

def reset():
    setup()

def radec_to_cartesian(ra, dec, distance):
    ra_rad = ra / 12 * math.pi  # Convert RA to radians
    dec_rad = dec / 180 * math.pi  # Convert DEC to radians
    
    x = distance * math.cos(ra_rad) * math.cos(dec_rad)
    y = distance * math.sin(ra_rad) * math.cos(dec_rad)
    z = distance * math.sin(dec_rad)
    
    return (x, y, z)

def cartesian_to_radec(x, y, z):
    # Compute distance (radius)
    distance = math.sqrt(x**2 + y**2 + z**2)

    ra = math.atan2(y, x)  # atan2 handles the quadrant issues
    ra = ra * 12 / math.pi  # Convert RA to degrees
    dec = math.atan2(z, math.sqrt(x**2 + y**2))  # atan2(z, r_xy)
    dec = dec * 180 / math.pi

    return(ra, dec, distance)

# Brightness is a size constraint
def brightness(vmag): 
    return 75 * math.pow(1.35, min(-vmag, 0.15))

### TODO
# To calculate relative ra dec from planet
# Subtract cartesian of planet from star, revert back to ra dec
def process_row(row, col):
    # Extract values from both row and col
    ra_star = float(row['ra'])
    dec_star = float(row['dec'])
    dist_star = float(row['sy_dist'])
    
    ra_planet = float(col['ra'])
    dec_planet = float(col['dec'])
    dist_planet = float(col['sy_dist'])
    
    # Convert to cartesian
    x_star, y_star, z_star = radec_to_cartesian(ra_star, dec_star, dist_star)
    x_planet, y_planet, z_planet = radec_to_cartesian(ra_planet, dec_planet, dist_planet)

    # Subtract vectors
    x_final = x_star + x_planet
    y_final = y_star + y_planet
    z_final = z_star + z_planet
    
    # Convert back
    ra_final, dec_final, dist_final = cartesian_to_radec(x_final, y_final, z_final)
    
    return pd.Series({'sy_name': row['sy_name'], 'ra': ra_final, 'dec': dec_final, 'sy_dist': dist_final, 'sy_vmag': row['sy_vmag']})

# TESTING ONLY
@app.route('/radec_cartesian', methods=['GET'])
def radec_cartesian():
    ra = float(request.args.get('ra'))
    dec = float(request.args.get('dec'))
    distance = float(request.args.get('distance'))
    ret = radec_to_cartesian(ra, dec, distance)
    return jsonify({'x': ret[0], 'y': ret[1], 'z': ret[2]})

# TESTING ONLY
@app.route('/cartesian_radec', methods=['GET'])
def cartesian_radec():
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    z = float(request.args.get('z'))
    ret = cartesian_to_radec(x, y, z)
    return jsonify({'ra': ret[0], 'dec': ret[1], 'distance': ret[2]})

@app.route('/selector')
def selector(): 
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query(f"SELECT * from planets", conn)
    df_sample = df.sample(n=1000).sort_index(ascending=True)
    ret = df_sample[['pl_name', 'pl_bmasse', 'pl_orbper', 'pl_eqt', 'pl_orbeccen', 'pl_insol', 'st_spectype', 'st_teff', 'st_rad', 'ra', 'dec', 'sy_dist']]
    global sample
    sample = ret
    return ret.to_json(orient="index")

@app.route('/render')
def render():
    selected_idx = int(request.args.get('index'))
    global sample
    col = sample.iloc[selected_idx]
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query(f"SELECT * from stars", conn)
    df = df[['sy_name', 'ra', 'dec', 'sy_dist', 'sy_vmag']]
    df = df.apply(process_row, axis=1, col=col)
    df_sorted = df.sort_values(by='sy_dist', ascending=True)
    df_sorted = df_sorted.head(1000)
    print(df_sorted)
    df_sorted['sy_vmag'] = df_sorted['sy_vmag'].apply(brightness)
    planet_json = col.to_json()
    stars_json = df_sorted.to_json(orient='index')
    return jsonify({"planet": planet_json, "stars": stars_json})

@app.route('/generate_text')
def generate_text():
    yapper = TEXT_BOT
    message = request.args.get('message')
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": yapper},
            {
                "role": "user",
                "content": message
            }
        ]
    )
    return jsonify({'message': completion.choices[0].message})

@app.route('/generate_image')
def generate_image():
    yapper = IMAGE_BOT
    url = request.args.get('url')
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": yapper},
                {
                "type": "image_url",
                "image_url": {
                    "url": url,
                },
                },
            ],
            }
        ],
        max_tokens=300,
    )
    return jsonify({'message': completion.choices[0].message})

@app.route('/')
def home():
    return "Welcome to the Exosky backend!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional DB reset")
    parser.add_argument('--reset', action='store_true', help="Reset the DB")
    args = parser.parse_args()
    if args.reset:
        print("DB reset initated")
        reset()
    app.run(debug=True)
