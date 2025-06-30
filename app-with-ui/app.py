from flask import Flask, render_template, request
from gaya_belajar_flask import LearningStyleClassifier
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Inisialisasi Global ---
# Membuat satu instance dari classifier untuk digunakan di seluruh aplikasi
classifier = LearningStyleClassifier()

# Path ke file CSV Anda
CSV_FILE_PATH = "SURVEY_GAYA_BELAJAR.csv"

# Memuat dan memproses data HANYA SEKALI saat aplikasi pertama kali dijalankan
with app.app_context():
    if os.path.exists(CSV_FILE_PATH):
        classifier.initialize_data(CSV_FILE_PATH)
    else:
        print(f"FATAL ERROR: File data '{CSV_FILE_PATH}' tidak ditemukan!")
        # Anda bisa menambahkan logic untuk menghentikan aplikasi jika file tidak ada

@app.route('/')
def index():
    """Menampilkan halaman utama dengan form pencarian."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Memproses permintaan pencarian dari form dan menampilkan hasilnya."""
    query = request.form.get('query', '')
    student_data = classifier.find_student_by_name(query)
    
    # Menambahkan emoji untuk ditampilkan di template hasil pencarian
    if student_data and 'Gaya_Belajar_Teridentifikasi' in student_data:
        style = student_data['Gaya_Belajar_Teridentifikasi']
        emoji_map = {'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲'}
        student_data['emoji'] = emoji_map.get(style, '❓')

    return render_template('results.html', student=student_data, query=query)

@app.route('/dashboard')
def dashboard():
    """Menampilkan halaman dashboard dengan beberapa grafik dan data."""
    # Data untuk ringkasan persentase
    percentages = classifier.get_style_percentages()

    # Data untuk grafik 1: Jumlah total per gaya belajar
    overall_dist = classifier.get_style_distribution()
    
    # Data untuk grafik 2: Berdasarkan jenis kelamin
    gender_data = classifier.get_distribution_by_gender()
    
    # Data untuk grafik 3: Berdasarkan semester
    semester_data = classifier.get_distribution_by_semester()

    # Data untuk tabel
    all_students = classifier.get_all_students()

    return render_template('dashboard.html',
                           percentages=percentages,
                           overall_dist=overall_dist,
                           gender_data=gender_data,
                           semester_data=semester_data,
                           students=all_students)

if __name__ == '__main__':
    # host='0.0.0.0' agar bisa diakses dari perangkat lain di jaringan yang sama
    app.run(debug=True, host='0.0.0.0')