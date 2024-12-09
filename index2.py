import pandas as pd
from geopy.distance import geodesic

# --- 1. Import Dataset ---
userprofile = pd.read_csv('userprofile.csv')
usercuisine = pd.read_csv('usercuisine.csv')
chefmozcuisine = pd.read_csv('chefmozcuisine.csv')
geoplaces2 = pd.read_csv('geoplaces2.csv')
ratings_final = pd.read_csv('rating_final.csv')

# --- 2. Fungsi untuk Mencari Preferensi Makanan ---
def get_user_preferences(user_id, usercuisine):
    preferences = usercuisine[usercuisine['userID'] == user_id]['Rcuisine']
    return preferences.tolist()

# --- 3. Fungsi untuk Mencari Restoran Sesuai Preferensi ---
def get_matching_restaurants(preferences, chefmozcuisine):
    matching_restaurants = chefmozcuisine[chefmozcuisine['Rcuisine'].isin(preferences)]
    return matching_restaurants['placeID'].unique()

# --- 4. Fungsi untuk Mencari Restoran Terdekat ---
def find_nearest_restaurants(user_location, matching_restaurants, geoplaces2, chefmozcuisine, top_n=5):
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

# --- 5. Fungsi untuk Menambahkan Rating ---
def add_ratings(nearest_restaurants, ratings_final):
    nearest_restaurants = nearest_restaurants.copy()
    nearest_restaurants['avg_rating'] = nearest_restaurants['placeID'].apply(
        lambda place_id: ratings_final[ratings_final['placeID'] == place_id]['rating'].mean()
    )
    return nearest_restaurants

# --- 6. Main Logic ---
def recommend_restaurants(user_id, top_n=5):
    # Ambil lokasi pengguna
    user_row = userprofile[userprofile['userID'] == user_id]
    if user_row.empty:
        print(f"User {user_id} tidak ditemukan!")
        return None
    user_location = (user_row.iloc[0]['latitude'], user_row.iloc[0]['longitude'])

    # Ambil preferensi makanan pengguna
    preferences = get_user_preferences(user_id, usercuisine)
    if not preferences:
        print(f"User {user_id} tidak memiliki preferensi makanan!")
        return None

    # Cari restoran sesuai preferensi
    matching_restaurants = get_matching_restaurants(preferences, chefmozcuisine)

    # Cari restoran terdekat
    nearest_restaurants = find_nearest_restaurants(user_location, matching_restaurants, geoplaces2, chefmozcuisine, top_n)

    # Tambahkan rating
    final_recommendations = add_ratings(nearest_restaurants, ratings_final)

    return final_recommendations

# --- 7. Input User ---
user_id = input("Masukkan ID User (contoh: U1001): ").strip()
recommendations = recommend_restaurants(user_id)

# --- 8. Output ---
if recommendations is not None:
    if 'Rcuisine' in recommendations.columns:
        print("\nRekomendasi Restoran Terdekat:")
        print(recommendations[['name', 'address', 'Rcuisine', 'distance', 'avg_rating']])
    else:
        print("\nKolom 'Rcuisine' tidak ditemukan. Berikut adalah hasil rekomendasi:")
        print(recommendations.head())
    
    recommendations.to_csv(f'recommendations_for_{user_id}.csv', index=False)
    print("\nHasil rekomendasi telah disimpan dalam CSV!")
