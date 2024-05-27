# Görüntü İşleme Yöntemleri Kullanarak El ve Parmak Rahatsızlıklarının Tespiti Projesi

Bu proje, görüntü işleme yöntemlerini kullanarak el ve parmak rahatsızlıklarının teşhis ve tedavi doğruluğunu artırmayı amaçlamaktadır. MediaPipe ve OpenCV kullanılarak el eklemleri tespit edilir, parmak açıları hesaplanır ve el durumu değerlendirilir. Ayrıca analiz edilen verilere dayanarak detaylı raporlar oluşturulur.

## Proje Hakkında

Bu proje, el ve parmak rahatsızlıklarının teşhisinde ve tedavisinde doğruluğu artırmayı hedeflemektedir. Görüntü işleme teknikleri ve makine öğrenimi algoritmaları kullanılarak:
- **Rahatsızlıkların otomatik tespiti** 
- **Tedavi süreçlerinin izlenmesi** 
- **Tedavi etkinliğinin değerlendirilmesi sağlanacaktır** 

## Özellikler

- **El Durumu Tespiti:** Başparmak, işaret parmağı, orta parmak, yüzük parmağı ve serçe parmağı arasındaki açılar hesaplanarak el durumu belirlenir.
- **Parmak Açılarının Hesaplanması:** Her bir parmak için eklemler arasındaki açılar hesaplanır ve analiz edilir.
- **Raporlama:** Her bir analiz için ayrıntılı raporlar oluşturulur ve belirtilen klasöre kaydedilir.
- **Kullanıcı Dostu Arayüz:** Kullanımı kolay arayüz, sağlık uzmanlarının ve kullanıcıların verileri kolayca anlamasını sağlar.

## Kurulum

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin:
### 1. Depoyu Klonlayın:

    ```bash
    git clone https://github.com/kullanici_adi/proje_adi.git
    cd proje_adi
    ```
### 2. Gerekli Bağımlılıkları Yükleyin:

    ```bash
     pip install -r requirements.txt
    ```

## Kullanım

Projeyi çalıştırmak için aşağıdaki komutları kullanabilirsiniz:
### 1. Ana Scripti Çalıştırın:

    ```bash
     python main.py
    ```
    
## Fonksiyonlar

Projede kullanılan temel fonksiyonlar aşağıda açıklanmıştır:
- **calculate_angle(coords1, coords2, coords3):** Üç nokta arasındaki açıyı hesaplar.
- **get_landmark_coords(landmark, img_width, img_height):** Landmark koordinatlarını alır.
- **calculate_thumb_angle(hand_landmarks, img_width, img_height):** Başparmak açısını hesaplar.
- **calculate_index_finger_angle(hand_landmarks, img_width, img_height):** İşaret parmağı açısını hesaplar.
- **calculate_middle_finger_angle(hand_landmarks, img_width, img_height):** Orta parmak açısını hesaplar.
- **calculate_ring_finger_angle(hand_landmarks, img_width, img_height):** Yüzük parmağı açısını hesaplar.
- **calculate_pinky_finger_angle(hand_landmarks, img_width, img_height):** Serçe parmağı açısını hesaplar.
- **detect_hand_status(thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle):** El durumunu tespit eder.
- **write_to_report(report_file, thumb_angle, index_angle, mid_angle, ring_angle, pinky_angle, hand_status):** Rapor dosyasına yazma işlemini yapar.

## Rapor Oluşturma
Program çalışırken 'n' tuşuna basıldığında yeni bir rapor oluşturulur. Raporlar " C:/Raporlar " klasöründe saklanır ve her rapor oluşturulduğu zamana göre adlandırılır.

 
