import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class LearningStyleClassifier:
    def __init__(self):
        self.df = None

    def load_data(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
            # Membersihkan data semester, mengubahnya menjadi integer jika memungkinkan
            if 'Semester' in self.df.columns:
                self.df['Semester'] = pd.to_numeric(self.df['Semester'], errors='coerce').dropna().astype(int)
            return True
        except Exception as e:
            print(f"❌ Error saat memuat data: {e}")
            return False

    def identify_learning_styles(self):
        if self.df is None: return

        learning_styles = []
        visual_keywords = ['papan tulis', 'membaca', 'gambar', 'petunjuk', 'menonton', 'melihat', 'visual', 'seni rupa', 'ekspresi wajah', 'mencoret', 'video', 'diagram', 'grafik', 'ilustrasi', 'foto', 'tulisan', 'warna']
        auditory_keywords = ['mendengar', 'suara', 'musik', 'diskusi', 'berbicara', 'nada', 'radio', 'mengucapkan', 'keras', 'penjelasan', 'ceramah', 'podcast', 'audio', 'dialog', 'percakapan']
        kinesthetic_keywords = ['praktek', 'praktikum', 'olahraga', 'fisik', 'gerakan', 'berjalan', 'mengetuk', 'demonstrasi', 'tangan', 'berolahraga', 'mencoba', 'melakukan', 'pengalaman', 'simulasi', 'aktif']
        question_cols = [col for col in self.df.columns if col not in ['Timestamp', 'Email', 'Nama', 'Asal Kampus', 'Prodi', 'Tanggal Lahir', 'Jenis Kelamin', 'Semester', 'Email Address']]

        for _, row in self.df.iterrows():
            combined_response = ' '.join([str(row[col]).lower() for col in question_cols if pd.notna(row[col])])
            scores = {
                'Visual': sum(1 for k in visual_keywords if k in combined_response),
                'Auditory': sum(1 for k in auditory_keywords if k in combined_response),
                'Kinesthetic': sum(1 for k in kinesthetic_keywords if k in combined_response)
            }
            dominant_style = max(scores, key=scores.get) if max(scores.values()) > 0 else 'Visual'
            learning_styles.append(dominant_style)
        
        self.df['Gaya_Belajar_Teridentifikasi'] = learning_styles
        print("✅ Identifikasi gaya belajar selesai.")

    def initialize_data(self, file_path):
        print("🚀 Memulai inisialisasi data...")
        if self.load_data(file_path):
            self.identify_learning_styles()
            print(f"✅ Data siap! Total {len(self.df)} mahasiswa diproses.")
        else:
            print("❌ Gagal melakukan inisialisasi data.")

    def find_student_by_name(self, name):
        if self.df is None or not name or name.strip() == "": return None
        result_df = self.df[self.df['Nama'].str.contains(name, case=False, na=False)]
        return result_df.iloc[0].to_dict() if not result_df.empty else None

    def get_all_students(self):
        return self.df.fillna('').to_dict('records') if self.df is not None else []

    # --- FUNGSI BARU UNTUK DASHBOARD ---
    def get_style_distribution(self):
        if self.df is None: return {}
        return self.df['Gaya_Belajar_Teridentifikasi'].value_counts().to_dict()

    def get_style_percentages(self):
        counts = self.get_style_distribution()
        total = sum(counts.values())
        if total == 0: return {}
        return {style: round((count / total) * 100, 1) for style, count in counts.items()}

    def _prepare_chart_data(self, crosstab_df):
        """Helper untuk mengubah data crosstab menjadi format Chart.js"""
        # Memastikan urutan gaya belajar konsisten
        style_order = ['Visual', 'Auditory', 'Kinesthetic']
        crosstab_df = crosstab_df.reindex(columns=style_order).fillna(0)
        
        chart_data = {
            "labels": crosstab_df.index.astype(str).tolist(),
            "datasets": []
        }
        colors = {
            'Visual': 'rgba(54, 162, 235, 0.8)',
            'Auditory': 'rgba(255, 99, 132, 0.8)',
            'Kinesthetic': 'rgba(75, 192, 192, 0.8)'
        }
        for style in style_order:
            if style in crosstab_df.columns:
                chart_data["datasets"].append({
                    "label": style,
                    "data": crosstab_df[style].tolist(),
                    "backgroundColor": colors.get(style)
                })
        return chart_data

    def get_distribution_by_gender(self):
        if self.df is None or 'Jenis Kelamin' not in self.df.columns: return {}
        ct = pd.crosstab(self.df['Jenis Kelamin'], self.df['Gaya_Belajar_Teridentifikasi'])
        return self._prepare_chart_data(ct)

    def get_distribution_by_semester(self):
        if self.df is None or 'Semester' not in self.df.columns: return {}
        # Mengurutkan semester
        df_sorted = self.df.sort_values('Semester')
        ct = pd.crosstab(df_sorted['Semester'], df_sorted['Gaya_Belajar_Teridentifikasi'])
        return self._prepare_chart_data(ct)