from flask import Flask, request, jsonify
import math 
from setup import setup
import argparse
import sqlite3
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from mimetypes import guess_type
from flask_cors import CORS

# Load .env
load_dotenv()

### GLOBALS
global sample

### CONSTANTS
TEXT_BOT = """You are telling an awe-inspiring story about the stars in the night sky."""
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NO_NANS = ['ra', 'dec', 'sy_dist', 'sy_vmag', 'sy_bmag']
IS_RESET = False
UPLOAD_FOLDER = "./uploads"

### INIT MODULES
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
client = OpenAI(api_key=OPENAI_API_KEY)

def reset():
    setup()

def delete_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.unlink(file_path)

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

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
    
    return pd.Series({
        'sy_name': row['sy_name'],
        'ra': ra_final,
        'dec': dec_final,
        'sy_dist': dist_final,
        'sy_vmag': row['sy_vmag'],
        'sy_bmag': row['sy_bmag'],
        'st_lum': row['st_lum'],
        'st_mass': row['st_mass'],
        'st_teff': row['st_teff']
    })

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
    x = request.args.get('x', type=float)
    y = request.args.get('y', type=float)
    z = request.args.get('z', type=float)
    ret = cartesian_to_radec(x, y, z)
    return jsonify({'ra': ret[0], 'dec': ret[1], 'distance': ret[2]})

@app.route('/selector')
def selector(): 
    max_size = request.args.get('max', default=None, type=int)
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query(f"SELECT * from planets", conn)
    ret = df[['pl_name', 'pl_bmasse', 'pl_orbincl', 'pl_eqt', 'st_teff', 'st_lum', 'st_vsin', 'ra', 'dec', 'sy_dist', 'sy_bmag', 'sy_vmag']]
    ret = ret.dropna(subset=NO_NANS)
    ret = ret.drop_duplicates(subset='pl_name', keep='first')
    if max_size: 
        ret = ret.sample(n=max_size).sort_index(ascending=True)
    ret.to_sql('sample', conn, if_exists='replace', index=True)
    sample = pd.read_sql_query(f'SELECT * FROM sample', conn)
    return ret.to_json(orient="index")

@app.route('/render')
def render():
    max_size = request.args.get('max', default=None, type=int)
    selected_idx = request.args.get('index', type=int)
    conn = sqlite3.connect('data.db')
    sample = pd.read_sql_query(f'SELECT * FROM sample', conn)
    col = sample[sample['index'] == selected_idx]
    df = pd.read_sql_query(f"SELECT * from stars", conn)
    if max_size:
        df = df.sample(n=max_size).sort_index(ascending=True)
    df = df[['sy_name', 'ra', 'dec', 'sy_dist', 'sy_vmag', 'sy_bmag', 'st_lum', 'st_mass', 'st_teff']]
    df = df.dropna(subset=NO_NANS)
    df = df.drop_duplicates(subset='sy_name', keep='first')
    df = df.apply(process_row, axis=1, col=col)
    df_sorted = df.sort_values(by='sy_dist', ascending=True)
    if max_size:
        df_sorted = df_sorted.head(max_size)
    planet_json = col.to_json()
    stars_json = df_sorted.to_json(orient='index')
    return jsonify({"planet": planet_json, "stars": stars_json})

@app.route('/extra')
def extra():
    max_size = request.args.get('max', default=None, type=int)
    conn = sqlite3.connect('data.db')
    df = pd.read_sql_query(f"SELECT * from koi", conn)
    if max_size():
        df = df.sample(n=1000).sort_index(ascending=True)
    return df.to_json(orient='index')

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
    return jsonify({'message': completion.choices[0].message.content})

@app.route('/generate_image')
def generate_image():
    uploads_folder = './uploads'
    image_file = None
    for filename in os.listdir(uploads_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):  # Add other extensions if needed
            image_file = os.path.join(uploads_folder, filename)
            break  # Assuming there's only one image
    if not image_file:
        return jsonify({'error': 'No image found in the uploads folder.'}), 400
    data_url = local_image_to_data_url(image_file)
    #url = request.args.get('url')
    city = request.args.get('city')
    country = request.args.get('country')
    yapper = f"""
    Given this fictional constellation image, generate a name and an engaging story, considering cultural views from {city}, {country}
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": yapper},
                {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                },
                },
            ],
            }
        ]
    )
    return jsonify({'message': completion.choices[0].message.content})

@app.route('/generate_image_b64')
def generate_image_b64():
    b64 = request.args.get('b64')
    city = request.args.get('city')
    country = request.args.get('country')
    yapper = f"""
    Given this fictional constellation image, generate a name and an engaging story, considering cultural views from {city}, {country}
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": yapper},
                {"type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                }
                }
            ]}
        ]
    )
    return jsonify({'message': completion.choices[0].message.content})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    delete_images(app.config['UPLOAD_FOLDER'])
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    return jsonify({'message': 'File uploaded successfully', 'path': filepath})

@app.route('/')
def home():
    return "Welcome to the Exosky backend!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional DB reset")
    parser.add_argument('--reset', action='store_true', help="Reset the DB")
    args = parser.parse_args()
    if args.reset and not IS_RESET:
        print("DB reset initated")
        reset()
        is_reset = True
    app.run(host="0.0.0.0", debug=True, port=5000)
