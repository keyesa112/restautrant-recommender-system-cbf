from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from geopy.distance import geodesic

# --- 1. Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

# --- 2. Import Dataset ---
userprofile = pd.read_csv('userprofile.csv')
usercuisine = pd.read_csv('usercuisine.csv')
chefmozcuisine = pd.read_csv('chefmozcuisine.csv')
geoplaces2 = pd.read_csv('geoplaces2.csv')
ratings_final = pd.read_csv('rating_final.csv')

# --- 3. Fungsi untuk Mencari Preferensi Makanan ---
def get_user_preferences(user_id):
    preferences = usercuisine[usercuisine['userID'] == user_id]['Rcuisine']
    return preferences.tolist()

# --- 4. Fungsi untuk Mencari Restoran Sesuai Preferensi ---
def get_matching_restaurants(preferences):
    matching_restaurants = chefmozcuisine[chefmozcuisine['Rcuisine'].isin(preferences)]
    return matching_restaurants['placeID'].unique()

# --- 5. Fungsi untuk Mencari Restoran Terdekat ---
def find_nearest_restaurants(user_location, matching_restaurants, top_n=5):
    # Ambil data restoran yang sesuai dengan `placeID`
    filtered_restaurants = geoplaces2[geoplaces2['placeID'].isin(matching_restaurants)].copy()
    filtered_restaurants['distance'] = filtered_restaurants.apply(
        lambda row: geodesic(user_location, (row['latitude'], row['longitude'])).km, axis=1
    )
    
    # Gabungkan dengan Rcuisine dari chefmozcuisine
    filtered_restaurants = filtered_restaurants.merge(
        chefmozcuisine[['placeID', 'Rcuisine']],
        on='placeID',
        how='left'
    )
    
    return filtered_restaurants.sort_values('distance').head(top_n)

# --- 6. Fungsi untuk Menambahkan Rating ---
def add_ratings(nearest_restaurants):
    nearest_restaurants = nearest_restaurants.copy()
    nearest_restaurants['avg_rating'] = nearest_restaurants['placeID'].apply(
        lambda place_id: ratings_final[ratings_final['placeID'] == place_id]['rating'].mean()
    )
    return nearest_restaurants

# --- 7. Main Logic ---
def recommend_restaurants(user_id, top_n=5):
    # Ambil lokasi pengguna
    user_row = userprofile[userprofile['userID'] == user_id]
    if user_row.empty:
        return None, f"User {user_id} tidak ditemukan!"
    
    user_location = (user_row.iloc[0]['latitude'], user_row.iloc[0]['longitude'])

    # Ambil preferensi makanan pengguna
    preferences = get_user_preferences(user_id)
    if not preferences:
        return None, f"User {user_id} tidak memiliki preferensi makanan!"

    # Cari restoran sesuai preferensi
    matching_restaurants = get_matching_restaurants(preferences)

    # Cari restoran terdekat
    nearest_restaurants = find_nearest_restaurants(user_location, matching_restaurants, top_n)

    # Tambahkan rating
    final_recommendations = add_ratings(nearest_restaurants)

    return final_recommendations, None

# --- 8. Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/booking')
def booking():
    return render_template('booking.html')

@app.route('/input_page', methods=['GET', 'POST'])
def input_page():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        return redirect(url_for('recommend', user_id=user_id))
    return render_template('input_page.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    user_id = request.args.get('user_id')  # Ambil user_id dari GET parameter
    if not user_id:
        return render_template('recommend.html', message="User ID tidak valid.", recommendations=None)

    # Panggil fungsi untuk mendapatkan rekomendasi
    recommendations, error_message = recommend_restaurants(user_id)
    if error_message:
        return render_template('recommend.html', message=error_message, recommendations=None)

    # Jika rekomendasi ditemukan
    return render_template('recommend.html', recommendations=recommendations.to_dict(orient='records'))

# --- 9. Menjalankan Aplikasi ---
if __name__ == '__main__':
    app.run(debug=True)
