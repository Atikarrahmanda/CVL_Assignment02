# Plat Detection using Template Matching

Tujuan utama dari tugas ini adalah mendeteksi lokasi plat nomor mobil secara otomatis, kemudian mengevaluasi hasilnya menggunakan metrik Akurasi (Accuracy) dan Intersection over Union (IoU).

## 1. Input Data
- Dataset berisi beberapa 5 gambar mobil yang diperoleh dari kaggle
  <img width="1512" height="458" alt="image" src="https://github.com/user-attachments/assets/ac1d3f02-246f-4829-8325-8a4c33f6a322" />

  
- Satu gambar plat digunakan sebagai **template** yang akan dicocokkan dengan seluruh gambar pada dataset.
![template (1)](https://github.com/user-attachments/assets/3b9b3535-8eb9-4519-9bce-a360e14ea41c)

## 2. Langkah-Langkah Proses
### a. Menentukan posisi ground truth
Posisi Ground truth ditentukan secara manual
 ```python
ground_truth = {
    "image1.jpg": (1092, 1740, 708, 168),
    "image2.jpg": (1062, 1860, 828, 174),
    "image3.jpg": (1110, 1818, 750, 204),
    "image4.jpg": (356, 538, 318, 108),
    "image5.jpg": (668, 610, 727, 142),
}
```
<img width="727" height="997" alt="image" src="https://github.com/user-attachments/assets/e44dc1c1-03c6-4893-8200-2f26c956e1a9" />

### b. Membaca gambar dan template, lalu mengubah keduanya menjadi grayscale untuk memudahkan perhitungan korelasi.
      Gambar template (contoh plat nomor) dan citra mobil dibaca kemudian diubah menjadi grayscale agar proses pencocokan lebih efisien dan tidak dipengaruhi warna.
  ```python
template = cv2.imread(template_path, 0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```


### c. Melakukan multi-scale template matching dengan berbagai skala (0.5× hingga 1.5×) untuk mengatasi perbedaan ukuran plat pada tiap gambar.
Dilakukan pencocokan pada berbagai skala untuk menemukan ukuran template paling sesuai.
```python
scales = np.linspace(0.5, 1.5, 10)

for scale in scales:
    tW = int(template.shape[1] * scale)
    tH = int(template.shape[0] * scale)
    tpl_scaled = cv2.resize(template, (tW, tH))
    result = cv2.matchTemplate(gray, tpl_scaled, cv2.TM_CCOEFF_NORMED)
```
### d. Menentukan posisi dengan korelasi tertinggi (cv2.minMaxLoc)
Fungsi ini digunakan untuk menemukan posisi dengan nilai korelasi tertinggi dari hasil template matching, yang dianggap sebagai posisi plat nomor pada gambar.
```python
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
best_box = (max_loc[0], max_loc[1], tW, tH)
```

### e. Menggambar bounding box Ground Truth (merah) dan Prediksi (hijau)
Setelah posisi ditemukan, sistem menggambar dua kotak:
<img width="1798" height="1133" alt="image" src="https://github.com/user-attachments/assets/b5d16982-8848-4ac2-9279-8b4387985157" />

Merah → posisi ground truth (label manual)
Hijau → posisi hasil prediksi dari template matching
```python
# Kotak Ground Truth
cv2.rectangle(image, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 3)

# Kotak Prediksi
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
```
### f. Menghitung IoU (Intersection over Union)
Nilai IoU digunakan untuk mengukur seberapa besar tumpang tindih antara kotak prediksi dan ground truth.
Semakin besar nilai IoU, semakin akurat deteksi yang dilakukan.
```python
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou
```

### f. Menghitung Accuracy dan Rata-rata IoU sebagai evaluasi akhir
Akurasi dihitung dari jumlah gambar dengan IoU ≥ 0.5 dibandingkan total gambar, sedangkan rata-rata IoU menunjukkan performa keseluruhan sistem.
```python
correct = sum(1 for r in results if r["IoU"] >= 0.5)
accuracy = correct / len(results)
avg_iou = np.mean([r["IoU"] for r in results if r["IoU"] > 0])

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Average IoU : {avg_iou:.3f}")
```
<img width="624" height="285" alt="image" src="https://github.com/user-attachments/assets/489d3f3d-62a4-4a73-8822-7356f9cc5c85" />

##Analisis Hasil
Metode Template Matching berhasil mendeteksi posisi plat nomor pada sebagian besar gambar.
Performa terbaik dicapai saat ukuran dan orientasi plat mirip dengan template.
Penurunan akurasi terjadi pada gambar dengan pencahayaan berbeda atau perbedaan sudut kamera.
Dengan rata-rata IoU 0.63 dan akurasi 80%, metode ini berhasil digunakan untuk mendeteksi plat nomor mobil tanpa menggunakan model deteksi modern.
Pendekatan ini sederhana namun cukup efektif untuk dataset kecil dan kondisi terkontrol.


