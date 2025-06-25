import os
from gaya_belajar import SimpleLearningStyleClassifier

def menu():
    print("\n" + "="*50)
    print("📚 APLIKASI ANALISIS GAYA BELAJAR MAHASISWA")
    print("="*50)
    print("1. Tampilkan semua data mahasiswa")
    print("2. Tampilkan statistik gaya belajar (dari identifikasi kuesioner)")
    print("3. Tampilkan persentase gaya belajar berdasarkan Jenis Kelamin")
    print("4. Tampilkan persentase gaya belajar berdasarkan Semester")
    print("5. Tampilkan grafik distribusi gaya belajar")
    print("6. Latih dan Evaluasi Model Prediksi") # Menu baru untuk melatih dan mengevaluasi
    print("7. Rekomendasi Strategi Pembelajaran") # Menu baru untuk rekomendasi
    # print("8. Prediksi Gaya Belajar Mahasiswa Baru") # Menu baru untuk prediksi
    print("0. Keluar")
    print("="*50)

def main():
    classifier = SimpleLearningStyleClassifier()
    # Pastikan struktur folder Anda adalah:
    # project_folder/
    # ├── main.py
    # ├── gaya_belajar.py
    # └── data/
    #     └── SURVEY_GAYA_BELAJAR.csv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "SURVEY_GAYA_BELAJAR.csv")

    # Muat dan identifikasi gaya belajar di awal agar data siap untuk semua menu
    if not classifier.load_data(file_path):
        return # Keluar jika gagal memuat data

    classifier.identify_learning_styles() # Identifikasi gaya belajar setelah data dimuat

    while True:
        menu()
        pilihan = input("🔍 Pilih menu: ")

        if pilihan == "1":
            classifier.show_student_learning_styles()
        elif pilihan == "2":
            classifier.show_learning_style_counts()
        elif pilihan == "3":
            classifier.show_percentage_by_gender()
        elif pilihan == "4":
            classifier.show_percentage_by_semester()
        elif pilihan == "5":
            classifier.plot_learning_style_distribution()
        elif pilihan == "6":
            # Panggil metode train_prediction_model untuk melatih dan menyimpan akurasi
            # Kemudian panggil evaluate_model secara eksplisit
            _ = classifier.train_prediction_model() 
            classifier.evaluate_model()
        elif pilihan == "7":
            classifier.generate_recommendations()
        # elif pilihan == "8":
        #     # Contoh input untuk prediksi mahasiswa baru
        #     print("\nMasukkan data mahasiswa baru:")
        #     jenis_kelamin = input("Jenis Kelamin (Laki-laki/Perempuan): ")
        #     # Asumsi semester adalah angka, pastikan input valid
        #     try:
        #         semester = int(input("Semester (contoh: 4): "))
        #     except ValueError:
        #         print("❌ Input semester tidak valid. Harus berupa angka.")
        #         continue # Lanjut ke menu utama
            
        #     new_student_data = {'Jenis Kelamin': jenis_kelamin, 'Semester': semester}
        #     classifier.predict_new_student(new_student_data)
        elif pilihan == "0":
            print("👋 Terima kasih, keluar dari program.")
            break
        else:
            print("❌ Pilihan tidak dikenali. Silakan coba lagi.")

if __name__ == "__main__":
    main()