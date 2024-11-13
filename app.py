from flask import Flask, render_template, request
import os
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process  # Untuk menangani typo

app = Flask(__name__)

# Fungsi untuk membaca data dari PDF
def read_pdf(file_path):
    brands = []
    
    # Cek jika file tidak ada
    if not os.path.exists(file_path):
        print(f"File PDF tidak ditemukan di path: {file_path}")
        return brands
    
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Cek jika PDF kosong
            if len(reader.pages) == 0:
                print("File PDF kosong.")
                return brands
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            brand_name = ' '.join(parts[:-2])
                            status = parts[-2]
                            category = parts[-1]
                            brands.append((brand_name, status, category))
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca file PDF: {e}")
        return brands
    
    return brands

# Fungsi untuk mencocokkan merek dan memberikan rekomendasi
def check_brand(brand_name, brands):
    if len(brand_name) < 2:
        return "Brand tidak ditemukan.", None, None, None, None

    # Membuat DataFrame untuk pemrosesan
    df = pd.DataFrame(brands, columns=['Nama Brand', 'Status', 'Kategori'])
    
    # Menggunakan TF-IDF untuk menghitung kesamaan
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Nama Brand'])
    query_vec = vectorizer.transform([brand_name])
    
    # Menghitung cosine similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Mencari hasil terbaik
    best_match_index = similarity.argmax()
    best_score = similarity[best_match_index]

    if best_score >= 0.7:  # Threshold untuk kesamaan
        match = df.iloc[best_match_index]
        status = match['Status']
        category = match['Kategori']
        
        # Mencari rekomendasi hanya jika status adalah "boikot"
        recommendations = None
        if status.lower() == 'boikot':
            recommendations = df[(df['Status'] == "Tidak") & (df['Kategori'] == category)]['Nama Brand'].tolist()
        
        # Menentukan warna status
        status_color = 'boikot' if status.lower() == 'boikot' else 'aman'
        
        return f"{match['Nama Brand']}", status, recommendations, status_color, None
    else:
        # Fuzzy matching untuk menangani typo
        possible_match = process.extractOne(brand_name, df['Nama Brand'])
        
        if possible_match:
            matched_brand = possible_match[0]
            return "Brand tidak ditemukan.", None, None, None, matched_brand
        else:
            return "Brand tidak ditemukan.", None, None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    status = None
    recommendations = None
    status_color = None
    possible_match = None

    if request.method == 'POST':
        brand_name = request.form['brand_name']
        brands_data = read_pdf('static/Produk.pdf')
        
        # Cek jika PDF tidak ada atau tidak valid
        if not brands_data:
            result = "File PDF tidak valid atau tidak ditemukan."
        else:
            result, status, recommendations, status_color, possible_match = check_brand(brand_name, brands_data)

    return render_template('index.html', 
                           result=result, 
                           status=status, 
                           recommendations=recommendations, 
                           status_color=status_color, 
                           possible_match=possible_match)

# Hapus bagian app.run() untuk produksi
# if __name__ == '__main__':
#     app.run(debug=True)
