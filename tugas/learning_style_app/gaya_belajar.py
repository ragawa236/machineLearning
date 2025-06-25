import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Mengabaikan warning agar output lebih bersih
warnings.filterwarnings('ignore')

class SimpleLearningStyleClassifier:
    """
    Kelas untuk mendeteksi dan memprediksi gaya belajar mahasiswa
    berdasarkan data kuesioner dan karakteristik (Jenis Kelamin, Semester).
    """

    def __init__(self):
        self.df = None
        self.model = None
        self.encoders = {} # Untuk menyimpan LabelEncoder yang digunakan
        self.feature_columns = [] # Kolom fitur yang akan digunakan untuk ML
        self.model_accuracy = None # Untuk menyimpan akurasi model

    def load_data(self, file_path):
        """
        Memuat data dari file CSV yang berisi respons kuesioner.
        """
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Data berhasil dimuat! Total mahasiswa: {len(self.df)}")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' tidak ditemukan. Pastikan path file benar.")
            return False
        except Exception as e:
            print(f"❌ Error saat memuat data: {e}")
            return False

    def identify_learning_styles(self):
        """
        Mengidentifikasi gaya belajar dominan setiap mahasiswa
        berdasarkan analisis kata kunci dari jawaban kuesioner.
        """
        if self.df is None:
            print("❌ Data belum dimuat. Jalankan load_data() terlebih dahulu.")
            return

        learning_styles = []
        
        # Kolom yang akan diabaikan dari analisis kata kunci
        cols_to_ignore = ['Timestamp', 'Email', 'Nama', 'Asal Kampus', 'Prodi', 
                          'Tanggal Lahir', 'Jenis Kelamin', 'Semester', 'Email Address']

        # Dapatkan daftar kolom pertanyaan kuesioner
        question_cols = [col for col in self.df.columns if col not in cols_to_ignore]
        
        # Kata kunci untuk setiap gaya belajar
        visual_keywords = ['papan tulis', 'membaca', 'gambar', 'petunjuk', 'menonton', 
                           'melihat', 'visual', 'seni rupa', 'ekspresi wajah', 'mencoret', 
                           'video', 'diagram', 'grafik', 'ilustrasi', 'foto', 'tulisan', 'warna']
        auditory_keywords = ['mendengar', 'suara', 'musik', 'diskusi', 'berbicara', 
                             'nada', 'radio', 'mengucapkan', 'keras', 'penjelasan', 
                             'ceramah', 'podcast', 'audio', 'dialog', 'percakapan']
        kinesthetic_keywords = ['praktek', 'praktikum', 'olahraga', 'fisik', 'gerakan', 
                                'berjalan', 'mengetuk', 'demonstrasi', 'tangan', 'berolahraga',
                                'mencoba', 'melakukan', 'pengalaman', 'simulasi', 'aktif']

        for idx, row in self.df.iterrows():
            visual_score = 0
            auditory_score = 0
            kinesthetic_score = 0
            
            # Gabungkan semua respons pertanyaan menjadi satu string untuk analisis kata kunci
            responses = []
            for col in question_cols:
                if pd.notna(row[col]):
                    responses.append(str(row[col]).lower())
            
            combined_response = ' '.join(responses)
            
            # Hitung skor berdasarkan kata kunci
            visual_score += sum(1 for keyword in visual_keywords if keyword in combined_response)
            auditory_score += sum(1 for keyword in auditory_keywords if keyword in combined_response)
            kinesthetic_score += sum(1 for keyword in kinesthetic_keywords if keyword in combined_response)
            
            scores = {
                'Visual': visual_score,
                'Auditory': auditory_score, 
                'Kinesthetic': kinesthetic_score
            }
            
            # Tentukan gaya belajar dominan
            dominant_style = max(scores, key=scores.get)
            
            # Heuristik sederhana jika semua skor sama (atau ada pilihan A/B/C literal dari kuesioner)
            # Ini asumsi jika kuesioner punya opsi A, B, C yang langsung direspon tanpa elaborasi
            # Anda bisa menyesuaikan atau menghapus bagian ini jika kuesioner hanya berisi esai/deskripsi
            if visual_score == auditory_score == kinesthetic_score:
                if 'a' in combined_response.split(): # Cek apakah ada 'a' sebagai jawaban tunggal
                    dominant_style = 'Visual'
                elif 'b' in combined_response.split(): # Cek apakah ada 'b' sebagai jawaban tunggal
                    dominant_style = 'Auditory'
                elif 'c' in combined_response.split(): # Cek apakah ada 'c' sebagai jawaban tunggal
                    dominant_style = 'Kinesthetic'
                else: # Jika tidak ada A/B/C tunggal dan skor sama, fallback ke visual sebagai default
                    dominant_style = 'Visual' # Atau bisa random, atau diserahkan ke user

            learning_styles.append(dominant_style)
        
        self.df['Gaya_Belajar_Teridentifikasi'] = learning_styles
        print("✅ Identifikasi gaya belajar dari kuesioner selesai!")
        return learning_styles
    
    def show_student_learning_styles(self):
        """Menampilkan nama mahasiswa beserta gaya belajarnya yang teridentifikasi."""
        if self.df is None or 'Gaya_Belajar_Teridentifikasi' not in self.df.columns:
            print("❌ Data atau gaya belajar belum teridentifikasi. Jalankan identify_learning_styles() terlebih dahulu.")
            return

        print("\n" + "="*70)
        print("📋 DAFTAR MAHASISWA DAN GAYA BELAJARNYA (Hasil Identifikasi Kuesioner)")
        print("="*70)
        
        for idx, row in self.df.iterrows():
            nama = row.get('Nama', f"Mahasiswa {idx+1}") # Ambil nama jika ada, kalau tidak pakai ID
            gaya_belajar = row['Gaya_Belajar_Teridentifikasi']
            kampus = row.get('Asal Kampus', 'N/A')
            
            emoji = {
                'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲'
            }
            
            print(f"{idx+1:2d}. {nama:<25} | {emoji.get(gaya_belajar, '❓')} {gaya_belajar:<12} | Kampus: {kampus}")
        
        print("="*70)
    
    def show_learning_style_counts(self):
        """Menampilkan jumlah dan persentase mahasiswa per gaya belajar."""
        if self.df is None or 'Gaya_Belajar_Teridentifikasi' not in self.df.columns:
            print("❌ Data atau gaya belajar belum teridentifikasi. Jalankan identify_learning_styles() terlebih dahulu.")
            return

        counts = self.df['Gaya_Belajar_Teridentifikasi'].value_counts()
        total = len(self.df)
        
        print("\n" + "="*60)
        print("📊 STATISTIK GAYA BELAJAR MAHASISWA (Hasil Identifikasi)")
        print("="*60)
        
        for style, count in counts.items():
            percentage = (count / total) * 100
            emoji = {'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲'}
            print(f"{emoji.get(style, '❓')} {style:<12}: {count:2d} mahasiswa ({percentage:.1f}%)")
        
        print(f"\n📈 Total Mahasiswa: {total}")
        print("="*60)
        
        return counts
    
    def show_percentage_by_gender(self):
        """Menampilkan persentase gaya belajar berdasarkan jenis kelamin."""
        if self.df is None or 'Gaya_Belajar_Teridentifikasi' not in self.df.columns:
            print("❌ Data atau gaya belajar belum teridentifikasi. Jalankan identify_learning_styles() terlebih dahulu.")
            return
        
        if 'Jenis Kelamin' not in self.df.columns or self.df['Jenis Kelamin'].isnull().all():
            print("❌ Data 'Jenis Kelamin' tidak tersedia.")
            return

        print("\n" + "="*70)
        print("👥 PERSENTASE GAYA BELAJAR BERDASARKAN JENIS KELAMIN")
        print("="*70)
        
        # Buat crosstab untuk analisis
        gender_style = pd.crosstab(self.df['Jenis Kelamin'], self.df['Gaya_Belajar_Teridentifikasi'], margins=True)
        
        # Hitung persentase
        gender_style_pct = pd.crosstab(self.df['Jenis Kelamin'], self.df['Gaya_Belajar_Teridentifikasi'], normalize='index') * 100
        
        emoji = {'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲'}
        
        for gender in gender_style.index[:-1]:  # Exclude 'All' row
            print(f"\n👤 {gender}:")
            total_gender = gender_style.loc[gender, 'All']
            
            for style in ['Visual', 'Auditory', 'Kinesthetic']:
                if style in gender_style.columns:
                    count = gender_style.loc[gender, style]
                    percentage = gender_style_pct.loc[gender, style]
                    print(f"   {emoji.get(style, '❓')} {style:<12}: {count:2d} mahasiswa ({percentage:.1f}%)")
            
            print(f"   📊 Total {gender}: {total_gender} mahasiswa")
        
        print("="*70)
        return gender_style, gender_style_pct
    
    def show_percentage_by_semester(self):
        """Menampilkan persentase gaya belajar berdasarkan semester."""
        if self.df is None or 'Gaya_Belajar_Teridentifikasi' not in self.df.columns:
            print("❌ Data atau gaya belajar belum teridentifikasi. Jalankan identify_learning_styles() terlebih dahulu.")
            return
        
        if 'Semester' not in self.df.columns or self.df['Semester'].isnull().all():
            print("❌ Data 'Semester' tidak tersedia.")
            return

        print("\n" + "="*70)
        print("🎓 PERSENTASE GAYA BELAJAR BERDASARKAN SEMESTER")
        print("="*70)
        
        # Buat crosstab untuk analisis
        semester_style = pd.crosstab(self.df['Semester'], self.df['Gaya_Belajar_Teridentifikasi'], margins=True)
        
        # Hitung persentase
        semester_style_pct = pd.crosstab(self.df['Semester'], self.df['Gaya_Belajar_Teridentifikasi'], normalize='index') * 100
        
        emoji = {'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲'}
        
        # Urutkan semester
        semesters = sorted([sem for sem in semester_style.index if sem != 'All'])
        
        for semester in semesters:
            print(f"\n📚 Semester {semester}:")
            total_semester = semester_style.loc[semester, 'All']
            
            for style in ['Visual', 'Auditory', 'Kinesthetic']:
                if style in semester_style.columns:
                    count = semester_style.loc[semester, style]
                    percentage = semester_style_pct.loc[semester, style]
                    print(f"   {emoji.get(style, '❓')} {style:<12}: {count:2d} mahasiswa ({percentage:.1f}%)")
            
            print(f"   📊 Total Semester {semester}: {total_semester} mahasiswa")
        
        print("="*70)
        return semester_style, semester_style_pct
    
    def plot_learning_style_distribution(self):
        """Menampilkan grafik distribusi gaya belajar dan hubungannya dengan fitur lain."""
        if self.df is None or 'Gaya_Belajar_Teridentifikasi' not in self.df.columns:
            print("❌ Data atau gaya belajar belum teridentifikasi. Jalankan identify_learning_styles() terlebih dahulu.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] # Warna konsisten untuk Visual, Auditory, Kinesthetic

        # Distribusi Gaya Belajar (Pie Chart)
        counts = self.df['Gaya_Belajar_Teridentifikasi'].value_counts()
        wedges, texts, autotexts = axes[0, 0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                                                  colors=colors, startangle=90, explode=(0.05, 0.05, 0.05),
                                                  textprops={'fontsize': 12, 'color': 'black'})
        axes[0, 0].set_title('📊 Distribusi Gaya Belajar Mahasiswa', fontsize=16, fontweight='bold', pad=20)
        for autotext in autotexts:
            autotext.set_color('white') # Warna teks persentase agar terlihat lebih jelas
            autotext.set_weight('bold')

        # Jumlah Mahasiswa per Gaya Belajar (Bar Chart)
        bars = axes[0, 1].bar(counts.index, counts.values, color=colors, alpha=0.9)
        axes[0, 1].set_title('📈 Jumlah Mahasiswa per Gaya Belajar', fontsize=16, fontweight='bold', pad=20)
        axes[0, 1].set_ylabel('Jumlah Mahasiswa', fontsize=12)
        axes[0, 1].set_xlabel('Gaya Belajar', fontsize=12)
        axes[0, 1].tick_params(axis='x', labelsize=11)
        axes[0, 1].tick_params(axis='y', labelsize=11)
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, height + 0.1,
                            f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Distribusi berdasarkan Jenis Kelamin (jika kolom ada)
        if 'Jenis Kelamin' in self.df.columns and not self.df['Jenis Kelamin'].isnull().all():
            gender_style = pd.crosstab(self.df['Jenis Kelamin'], self.df['Gaya_Belajar_Teridentifikasi'])
            gender_style.plot(kind='bar', ax=axes[1, 0], color=colors, rot=0, alpha=0.9)
            axes[1, 0].set_title('👥 Gaya Belajar berdasarkan Jenis Kelamin', fontsize=14, fontweight='bold', pad=15)
            axes[1, 0].set_xlabel('Jenis Kelamin', fontsize=12)
            axes[1, 0].set_ylabel('Jumlah Mahasiswa', fontsize=12)
            axes[1, 0].tick_params(axis='x', labelsize=11)
            axes[1, 0].tick_params(axis='y', labelsize=11)
            axes[1, 0].legend(title='Gaya Belajar', fontsize=10, title_fontsize=10)
        else:
            axes[1, 0].set_visible(False) # Sembunyikan subplot jika data tidak ada
            axes[1, 0].text(0.5, 0.5, 'Data "Jenis Kelamin" tidak tersedia',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[1, 0].transAxes, fontsize=12, color='gray')


        # Distribusi berdasarkan Semester (jika kolom ada)
        if 'Semester' in self.df.columns and not self.df['Semester'].isnull().all():
            # Pastikan semester diurutkan dengan benar jika numerik
            semester_style = pd.crosstab(self.df['Semester'], self.df['Gaya_Belajar_Teridentifikasi'])
            semester_style.plot(kind='bar', ax=axes[1, 1], color=colors, rot=0, alpha=0.9)
            axes[1, 1].set_title('🎓 Gaya Belajar berdasarkan Semester', fontsize=14, fontweight='bold', pad=15)
            axes[1, 1].set_xlabel('Semester', fontsize=12)
            axes[1, 1].set_ylabel('Jumlah Mahasiswa', fontsize=12)
            axes[1, 1].tick_params(axis='x', labelsize=11)
            axes[1, 1].tick_params(axis='y', labelsize=11)
            axes[1, 1].legend(title='Gaya Belajar', fontsize=10, title_fontsize=10)
        else:
            axes[1, 1].set_visible(False) # Sembunyikan subplot jika data tidak ada
            axes[1, 1].text(0.5, 0.5, 'Data "Semester" tidak tersedia',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[1, 1].transAxes, fontsize=12, color='gray')


        plt.tight_layout(pad=3.0)
        plt.suptitle('Analisis & Visualisasi Gaya Belajar Mahasiswa', fontsize=20, fontweight='bold', y=1.02)
        plt.show()
    
    def train_prediction_model(self):
        """
        Melatih model Random Forest untuk memprediksi gaya belajar
        berdasarkan fitur seperti Jenis Kelamin dan Semester.
        """
        if self.df is None or 'Gaya_Belajar_Teridentifikasi' not in self.df.columns:
            print("❌ Data atau gaya belajar belum teridentifikasi. Jalankan identify_learning_styles() terlebih dahulu.")
            return None
        
        print("\n🤖 Melatih model Machine Learning untuk prediksi...")
        
        # Kolom fitur yang akan digunakan untuk training model
        # Anda bisa menyesuaikan ini berdasarkan kolom yang relevan di CSV Anda
        self.feature_columns = ['Jenis Kelamin', 'Semester'] 
        
        # Pastikan kolom fitur ada di dataframe dan tidak kosong
        for col in self.feature_columns:
            if col not in self.df.columns or self.df[col].isnull().all():
                print(f"⚠️ Kolom '{col}' tidak ditemukan atau kosong. Model mungkin tidak optimal.")
                # Menghapus kolom dari feature_columns jika tidak ada data yang valid
                self.feature_columns = [f for f in self.feature_columns if f != col]

        if not self.feature_columns:
            print("❌ Tidak ada kolom fitur yang valid untuk melatih model. Batalkan pelatihan.")
            return None

        X = pd.DataFrame()
        for col in self.feature_columns:
            le = LabelEncoder()
            # Gunakan fillna('Unknown') untuk menangani nilai NaN pada kolom kategorikal
            X[col] = le.fit_transform(self.df[col].astype(str).fillna('Unknown')) 
            self.encoders[col] = le # Simpan encoder untuk prediksi di masa depan
        
        y = self.df['Gaya_Belajar_Teridentifikasi']
        
        # Memisahkan data latih dan data uji (70% latih, 30% uji)
        # Random_state digunakan agar hasil selalu sama setiap kali dijalankan
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Inisialisasi dan latih model Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluasi model pada data uji
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.model_accuracy = accuracy  # Simpan akurasi untuk digunakan nanti
        
        print(f"✅ Model berhasil dilatih.")
        print(f"📊 Akurasi Model (pada data uji): {accuracy:.2f} ({accuracy*100:.0f}%)")
        print("\n--- Laporan Klasifikasi ---")
        print(classification_report(y_test, y_pred))
        
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Prediksi Gaya Belajar')
        plt.ylabel('Gaya Belajar Sebenarnya')
        plt.show()

        return accuracy
    
    def evaluate_model(self):
        """Menampilkan evaluasi detail dari model yang telah dilatih."""
        if self.model is None:
            print("❌ Model belum dilatih. Jalankan train_prediction_model() terlebih dahulu.")
            return
        
        print("\n" + "="*60)
        print("🔍 EVALUASI DETAIL MODEL MACHINE LEARNING")
        print("="*60)
        
        if self.model_accuracy is not None:
            print(f"📊 Akurasi Model: {self.model_accuracy:.3f} ({self.model_accuracy*100:.1f}%)")
        
        # Tampilkan fitur yang digunakan
        print(f"🔧 Fitur yang digunakan: {', '.join(self.feature_columns)}")
        print(f"🎯 Kelas yang diprediksi: {', '.join(self.model.classes_)}")
        
        # Feature importance jika ada
        if hasattr(self.model, 'feature_importances_'):
            print("\n📈 Tingkat Kepentingan Fitur:")
            for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
                print(f"   • {feature}: {importance:.3f}")
        
        print("="*60)
    
    def predict_new_student(self, new_student_data):
        """
        Memprediksi gaya belajar untuk mahasiswa baru berdasarkan input.
        Args:
            new_student_data (dict): Dictionary berisi fitur mahasiswa baru,
                                     misal: {'Jenis Kelamin': 'Laki-laki', 'Semester': 4}
        """
        if self.model is None or not self.encoders:
            print("❌ Model belum dilatih atau encoder tidak tersedia. Jalankan train_prediction_model() terlebih dahulu.")
            return None
        
        print("\n🔮 PREDIKSI GAYA BELAJAR UNTUK MAHASISWA BARU")
        
        # Siapkan data input untuk prediksi
        input_data = pd.DataFrame([new_student_data])
        
        # Encode input menggunakan encoder yang sudah dilatih
        for col in self.feature_columns:
            if col in input_data.columns:
                try:
                    # Pastikan input kategori ada dalam kelas encoder
                    input_data[col] = self.encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    print(f"⚠️ Peringatan: Nilai '{new_student_data.get(col)}' untuk '{col}' tidak dikenal oleh model.")
                    # Fallback ke nilai default atau penanganan error lain
                    return None
            else:
                print(f"❌ Fitur '{col}' yang dibutuhkan model tidak ada di input. Tidak dapat memprediksi.")
                return None

        try:
            # Prediksi
            prediction_encoded = self.model.predict(input_data[self.feature_columns])[0]
            probabilities = self.model.predict_proba(input_data[self.feature_columns])[0]
            
            # Ubah kembali prediksi dari angka ke label gaya belajar
            prediction_style = self.model.classes_[prediction_encoded]
            
            # Tampilkan probabilitas untuk setiap kelas
            prob_dict = dict(zip(self.model.classes_, probabilities))
            
            print(f"👤 Jenis Kelamin: {new_student_data.get('Jenis Kelamin', 'N/A')}")
            print(f"📚 Semester: {new_student_data.get('Semester', 'N/A')}")
            print(f"📊 Prediksi Gaya Belajar: {prediction_style}")
            print(f"📈 Probabilitas:")
            for style, prob in prob_dict.items():
                emoji = {'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲'}
                print(f"   {emoji.get(style, '❓')} {style}: {prob:.2f}")
            
            return prediction_style, prob_dict
            
        except Exception as e:
            print(f"❌ Error dalam prediksi: {e}")
            return None
    
    def generate_recommendations(self):
        """
        Menghasilkan rekomendasi strategi pembelajaran berdasarkan
        distribusi gaya belajar mahasiswa yang teridentifikasi.
        """
        if self.df is None or 'Gaya_Belajar_Teridentifikasi' not in self.df.columns:
            print("❌ Data atau gaya belajar belum teridentifikasi. Jalankan identify_learning_styles() terlebih dahulu.")
            return

        print("\n" + "="*70)
        print("💡 REKOMENDASI STRATEGI PEMBELAJARAN BERDASARKAN DISTRIBUSI KELAS")
        print("="*70)
        
        recommendations = {
            'Visual': [
                "📊 Gunakan **diagram, grafik, dan infografis**.",
                "🎨 Buat **mind map** dan flowchart.", 
                "📺 Sediakan **video pembelajaran**.",
                "📝 Berikan handout dengan **visual menarik**.",
                "🖼️ Gunakan **gambar dan ilustrasi** dalam penjelasan."
            ],
            'Auditory': [
                "🎵 Integrasikan **musik atau suara** yang relevan dalam pembelajaran.",
                "💬 Perbanyak **diskusi kelompok** dan debat.",
                "🎙️ Gunakan **podcast** dan audio pembelajaran.", 
                "👥 Lakukan **presentasi lisan dan ceramah** yang interaktif.",
                "🔊 Berikan **penjelasan verbal** yang jelas dan ringkas."
            ],
            'Kinesthetic': [
                "🏃 Lakukan **aktivitas fisik** dan permainan edukatif.",
                "🔬 Perbanyak **praktikum dan eksperimen** langsung.",
                "🤝 Gunakan **role playing dan simulasi**.",
                "✋ Belajar sambil **bergerak** (misal: *walking meeting*, berdiri saat belajar).",
                "🛠️ Sediakan **alat peraga** dan materi yang bisa dimanipulasi."
            ]
        }
        
        # Dapatkan jumlah mahasiswa per gaya belajar
        counts = self.df['Gaya_Belajar_Teridentifikasi'].value_counts()
        
        for style, count in counts.items():
            emoji = {'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲'}
            print(f"\n{emoji.get(style, '❓')} {style.upper()} ({count} mahasiswa):")
            for rec in recommendations[style]:
                print(f"   • {rec}")
        
        print("="*70)
    
    def run_full_analysis(self, file_path):
        """
        Menjalankan seluruh alur analisis: memuat data, mengidentifikasi gaya belajar,
        menampilkan statistik, memvisualisasikan, melatih model, dan memberikan rekomendasi.
        """
        print("🚀 MEMULAI ANALISIS LENGKAP GAYA BELAJAR MAHASISWA")
        print("="*70)
        
        if not self.load_data(file_path):
            return # Hentikan jika data gagal dimuat
            
        self.identify_learning_styles()
        self.show_student_learning_styles()
        self.show_learning_style_counts()
        self.plot_learning_style_distribution()
        
        # Latih model prediksi dan dapatkan akurasinya
        accuracy = self.train_prediction_model()
        if accuracy is not None:
            print(f"\n**Akurasi Model Prediksi Keseluruhan: {accuracy*100:.0f}%**")
        
        self.generate_recommendations()
        
        print("\n✅ ANALISIS LENGKAP SELESAI!")

# --- Fungsi Utama untuk Menjalankan Kode ---
def main():
    classifier = SimpleLearningStyleClassifier()
    
    # Ganti 'SURVEY GAYA BELAJAR SISWA (Jawaban) - Form Responses 1.csv'
    # dengan path file CSV Anda yang sebenarnya
    file_path = "SURVEY GAYA BELAJAR SISWA (Jawaban) - Form Responses 1.csv"
    
    # Jalankan analisis lengkap
    classifier.run_full_analysis(file_path)
    
    # --- Contoh Penggunaan Prediksi Mahasiswa Baru ---
    print("\n" + "="*70)
    print("🔮 CONTOH PENGGUNAAN MODEL UNTUK PREDIKSI MAHASISWA BARU")
    print("="*70)
    
    # Pastikan data yang Anda masukkan sesuai dengan kolom yang digunakan untuk melatih model
    # yaitu 'Jenis Kelamin' (string) dan 'Semester' (integer)
    new_student_data_1 = {'Jenis Kelamin': 'Laki-laki', 'Semester': 4}
    classifier.predict_new_student(new_student_data_1)
    
    new_student_data_2 = {'Jenis Kelamin': 'Perempuan', 'Semester': 2}
    classifier.predict_new_student(new_student_data_2)
    
    new_student_data_3 = {'Jenis Kelamin': 'Laki-laki', 'Semester': 6}
    classifier.predict_new_student(new_student_data_3)


if __name__ == "__main__":
    main()