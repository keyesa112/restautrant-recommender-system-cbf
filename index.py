import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# --- 1. Import Dataset ---
ratings_final = pd.read_csv('rating_final.csv')
chefmozcuisine = pd.read_csv('chefmozcuisine.csv')
geoplaces2 = pd.read_csv('geoplaces2.csv')

# --- 2. Cek Dataset: ratings_final ---
print("Ratings Final Dataset:")
print(ratings_final.head())

# Cek data kosong
print("\nNull Values in Ratings Dataset:", ratings_final.isnull().sum())
ratings_final = ratings_final.dropna()

# --- Tambahan: Cek Outlier ---
# Menggunakan IQR untuk mendeteksi outlier pada kolom 'rating'
Q1 = ratings_final['rating'].quantile(0.25)
Q3 = ratings_final['rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nIQR Boundaries: {lower_bound} - {upper_bound}")

# Filter data untuk menghilangkan outlier
ratings_final = ratings_final[(ratings_final['rating'] >= lower_bound) & (ratings_final['rating'] <= upper_bound)]
print(f"Data setelah menghilangkan outlier: {ratings_final.shape[0]} baris")

# --- 3. Normalisasi Data ---
# Normalisasi rating menggunakan Min-Max Scaling
scaler = MinMaxScaler()
ratings_final['rating_normalized'] = scaler.fit_transform(ratings_final[['rating']])
print("\nRating setelah normalisasi:")
print(ratings_final[['rating', 'rating_normalized']].head())

# --- 4. Cek Dataset: chefmozcuisine ---
print("\nChefmozcuisine Dataset:")
print(chefmozcuisine.head())

# Cek data kosong
print("Null Values in Chefmozcuisine Dataset:", chefmozcuisine.isnull().sum())
chefmozcuisine = chefmozcuisine.dropna()

# --- 5. Cek Dataset: geoplaces2 ---
print("\nGeoplaces2 Dataset:")
print(geoplaces2.head())

# Cek data kosong
print("Null Values in Geoplaces2 Dataset:", geoplaces2.isnull().sum())
geoplaces2 = geoplaces2.dropna()

# --- 6. Merge Datasets ---
# Gabungkan ratings_final dengan chefmozcuisine berdasarkan 'placeID'
ratings_with_cuisine = pd.merge(ratings_final, chefmozcuisine, on='placeID', how='left')
ratings_with_places = pd.merge(ratings_with_cuisine, geoplaces2, on='placeID', how='left')

# --- 7. Membuat Profil Pengguna Berdasarkan Masakan ---
# Konversi kolom 'Rcuisine' ke string dan ganti NaN dengan string kosong
ratings_with_places['Rcuisine'] = ratings_with_places['Rcuisine'].fillna('').astype(str)

user_ratings = ratings_with_places.groupby('userID')['Rcuisine'].apply(lambda x: ' '.join(x))
user_profiles = pd.DataFrame(user_ratings).reset_index()

# Tampilkan profil pengguna
print("\nUser Profiles:")
print(user_profiles.head())

# --- 8. Vectorisasi Masakan Menggunakan TF-IDF ---
tfidf = TfidfVectorizer()
user_profiles_tfidf = tfidf.fit_transform(user_profiles['Rcuisine'])

# --- 9. Profil Restoran Berdasarkan Masakan ---
restaurant_profiles = chefmozcuisine.groupby('placeID')['Rcuisine'].apply(lambda x: ' '.join(x)).reset_index()
restaurant_profiles_tfidf = tfidf.transform(restaurant_profiles['Rcuisine'])

# --- 10. Matriks Korelasi ---
cosine_sim = cosine_similarity(user_profiles_tfidf, restaurant_profiles_tfidf)
corr_matrix = pd.DataFrame(cosine_sim, index=user_profiles['userID'], columns=restaurant_profiles['placeID'])

# --- 11. Menampilkan Matriks Korelasi ---
print("\nCorrelation Matrix:")
print(corr_matrix.head())

# --- 12. Fungsi Rekomendasi Restoran ---
def recommend_restaurants(user_id, corr_matrix, top_n=5):
    if user_id not in corr_matrix.index:
        print(f"User {user_id} tidak ditemukan!")
        return pd.DataFrame()

    user_corr = corr_matrix.loc[user_id]
    top_indices = user_corr.sort_values(ascending=False).head(top_n).index
    recommended_places = restaurant_profiles[restaurant_profiles['placeID'].isin(top_indices)]
    return recommended_places

# --- 13. Penggunaan Fungsi Rekomendasi ---
user_id = input("Masukkan ID User (contoh: U1001) untuk mendapatkan rekomendasi: ")
recommended_restaurants = recommend_restaurants(user_id, corr_matrix)

print("\nRekomendasi Restoran:")
print(recommended_restaurants)

# Simpan hasil rekomendasi ke CSV
recommended_restaurants.to_csv('recommended_restaurants.csv', index=False)
print("\nHasil rekomendasi telah disimpan dalam 'recommended_restaurants.csv'")
