#Import Library
import sys
import cv2
import os
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
from sklearn.svm import SVC

import numpy as np

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.pushButton3.clicked.connect(self.gambar3)
        self.verifikasibutton.clicked.connect(self.verifikasi)
        self.verifikasibutton_live.clicked.connect(self.verifikasilive)

    def gambar1(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)",
                                                        options=options)

        if self.file_path:
            self.Image = cv2.imread(self.file_path)  # Load the image
            self.displayImage(1)

    def gambar2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.file_path2, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)",
                                                         options=options)

        if self.file_path2:
            self.Image = cv2.imread(self.file_path2)  # Load the image
            self.displayImage(2)

    def gambar3(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.file_path3, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg *.png *.jpeg)",
                                                         options=options)

        if self.file_path3:
            self.Image = cv2.imread(self.file_path3)  # Load the image
            self.displayImage(3)

        clone = self.Image.copy()
        roi = cv2.selectROI("Select ROI", clone, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        x, y, w, h = roi
        self.cropped_image = self.Image[y:y + h, x:x + w]






    def verifikasi(self):
        def extract_signature_features(image):
            # Konversi ke citra ke skala abu-abu
            # grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




            def convert_to_grayscale(image):
                # Ambil dimensi citra
                height, width, channels = image.shape

                # Buat citra kosong dengan ukuran yang sama dalam skala keabuan
                grayscale = np.zeros((height, width), dtype=np.uint8)

                # Loop melalui setiap piksel pada citra
                for y in range(height):
                    for x in range(width):
                        # Ambil nilai intensitas piksel pada setiap kanal
                        blue = image[y, x, 0]
                        green = image[y, x, 1]
                        red = image[y, x, 2]

                        # Hitung nilai intensitas rata-rata piksel dalam skala keabuan
                        grayscale[y, x] = int(0.2989 * red + 0.5870 * green + 0.1140 * blue)

                return grayscale

            grayscale = convert_to_grayscale(image)
            cv2.imshow("grayscale", grayscale)






            # Binarisasi citra menggunakan metode Otsu
            # _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


            def binarize(grayscale):
                H, W = grayscale.shape[:2]

                # Manual Otsu's method
                threshold = np.mean(grayscale)  # Initial threshold guess

                for _ in range(10):  # Iteratively refine threshold
                    foreground = grayscale[grayscale <= threshold]
                    background = grayscale[grayscale > threshold]
                    mean_foreground = np.mean(foreground) if len(foreground) > 0 else 0
                    mean_background = np.mean(background) if len(background) > 0 else 0
                    new_threshold = 0.5 * (mean_foreground + mean_background)
                    if abs(threshold - new_threshold) < 0.5:
                        break
                    threshold = new_threshold

                # Binarization using threshold
                binary = np.zeros_like(grayscale)
                binary[grayscale > threshold] = 0
                binary[grayscale < threshold] = 255

                return binary

            binary = binarize(grayscale)
            cv2.imshow("biner", binary)





            # Mendapatkan kontur tanda tangan
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Mengambil kontur tanda tangan terbesar
            max_contour = max(contours, key=cv2.contourArea)



            # Gambar kontur pada citra grayscale
            image_with_contour = cv2.drawContours(binary.copy(), max_contour, -1, (0, 255, 0), 2)

            # Tampilkan citra dengan kontur menggunakan matplotlib
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image_with_contour, cv2.COLOR_BGR2RGB))
            plt.title('Detected Contour')

            # Menghitung fitur huMoments
            moments = cv2.moments(max_contour)
            hu_moments = cv2.HuMoments(moments).flatten()

            # Menampilkan fitur huMoments
            plt.subplot(1, 2, 2)
            plt.bar(range(len(hu_moments)), hu_moments)
            plt.xlabel('Hu Moment')
            plt.ylabel('Value')
            plt.title('Hu Moments')

            # Menampilkan plot
            plt.tight_layout()
            plt.show()

            return hu_moments


        def verify_signature(signature_features, trained_features, threshold):
            # Menghitung jarak Euclidean antara fitur tanda tangan yang akan diverifikasi dengan setiap tanda tangan yang telah dilatih
            nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nbrs.fit(trained_features)
            distances, indices = nbrs.kneighbors(signature_features.reshape(1, -1))

            # Membandingkan jarak dengan ambang batas
            if distances[0][0] < threshold:
                return True, (1 - distances[0][0] / threshold) * 100  # Menghitung persentase keakuratan
            else:
                return False, (1 - distances[0][0] / threshold) * 0  # Menghitung persentase keakuratan

        # # Mendata tanda tangan yang telah dilatih
        # trained_signatures = []
        # trained_signatures.append(extract_signature_features(cv2.imread(self.file_path)))
        # trained_signatures.append(extract_signature_features(cv2.imread(self.file_path2)))
        # # Tambahkan tanda tangan lainnya jika diperlukan
        #
        # # Menentukan ambang batas untuk verifikasi
        # verification_threshold = 1
        #
        # # Mendeteksi dan memperoleh fitur tanda tangan yang akan diverifikasi
        # signature_to_verify = cv2.imread(self.file_path3)  # Use the currently loaded image for verification
        # signature_features = extract_signature_features(signature_to_verify)

        # Inisialisasi daftar trained_signatures
        trained_signatures = []

        # Mendapatkan daftar file gambar tanda tangan dari folder data latih
        folder_path = "dataLatih"  # Ganti dengan path folder data latih Anda

        # Loop melalui setiap file dalam folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            signature_image = cv2.imread(file_path)

            # Ekstraksi fitur tanda tangan dari gambar
            signature_features = extract_signature_features(signature_image)

            # Tambahkan fitur tanda tangan ke dalam daftar trained_signatures
            trained_signatures.append(signature_features)



        # Menentukan ambang batas untuk verifikasi
        verification_threshold = 0.4


        signature_to_verify = self.cropped_image # Gunakan gambar yang saat ini dimuat untuk verifikasi
        signature_features = extract_signature_features(signature_to_verify)

        # Melakukan verifikasi tanda tangan
        verification_result, accuracy = verify_signature(signature_features, trained_signatures, verification_threshold)

        if verification_result:
            if accuracy > 70:
                print("Tanda tangan terverifikasi dengan tingkat keakuratan {:.2f}%.".format(accuracy))


                # Set teks output pada QLabel
                self.label_2.setText("Tanda tangan terverifikasi")
            else :
                print("Tanda tangan tidak terverifikasi.")

        else:
            print("Tanda tangan tidak terverifikasi.")

        # Menampilkan persentase keakuratan
        print("Accuracy: {:.2f}%".format(accuracy))

        self.label_3.setText(" {:.2f}%".format(accuracy))

    def verifikasilive(self):
        def extract_signature_features(image):
            # Konversi ke citra ke skala abu-abu
            # grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



            def convert_to_grayscale(image):
                # Ambil dimensi citra
                height, width, channels = image.shape

                # Buat citra kosong dengan ukuran yang sama dalam skala keabuan
                grayscale = np.zeros((height, width), dtype=np.uint8)

                # Loop melalui setiap piksel pada citra
                for y in range(height):
                    for x in range(width):
                        # Ambil nilai intensitas piksel pada setiap kanal
                        blue = image[y, x, 0]
                        green = image[y, x, 1]
                        red = image[y, x, 2]

                        # Hitung nilai intensitas rata-rata piksel dalam skala keabuan
                        grayscale[y, x] = int(0.2989 * red + 0.5870 * green + 0.1140 * blue)

                return grayscale

            grayscale = convert_to_grayscale(image)
            cv2.imshow("grayscale", grayscale)

            # Binarisasi citra menggunakan metode Otsu
            # _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            def binarize(grayscale):
                H, W = grayscale.shape[:2]

                # Manual Otsu's method
                # threshold = np.mean(grayscale)  # Initial threshold guess
                threshold = 180

                for _ in range(10):  # Iteratively refine threshold
                    foreground = grayscale[grayscale <= threshold]
                    background = grayscale[grayscale > threshold]
                    mean_foreground = np.mean(foreground) if len(foreground) > 0 else 0
                    mean_background = np.mean(background) if len(background) > 0 else 0
                    new_threshold = 0.5 * (mean_foreground + mean_background)
                    if abs(threshold - new_threshold) < 0.5:
                        break
                    threshold = new_threshold

                # Binarization using threshold
                binary = np.zeros_like(grayscale)
                binary[grayscale > threshold] = 0
                binary[grayscale < threshold] = 255

                return binary

            binary = binarize(grayscale)


            cv2.imshow('verifikasi', binary)
            cv2.imshow("biner", binary)



            # Mendapatkan kontur tanda tangan
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Mengambil kontur tanda tangan terbesar
            max_contour = max(contours, key=cv2.contourArea)



            # Gambar kontur pada citra grayscale
            image_with_contour = cv2.drawContours(binary.copy(), max_contour, -1, (0, 255, 0), 2)

            # Tampilkan citra dengan kontur menggunakan matplotlib
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image_with_contour, cv2.COLOR_BGR2RGB))
            plt.title('Detected Contour')

            # Menghitung fitur huMoments
            moments = cv2.moments(max_contour)
            hu_moments = cv2.HuMoments(moments).flatten()

            # Menampilkan fitur huMoments
            plt.subplot(1, 2, 2)
            plt.bar(range(len(hu_moments)), hu_moments)
            plt.xlabel('Hu Moment')
            plt.ylabel('Value')
            plt.title('Hu Moments')

            # Menampilkan plot
            plt.tight_layout()
            plt.show()

            return hu_moments

        def verify_signature(signature_features, trained_features, threshold):
            # Menghitung jarak Euclidean antara fitur tanda tangan yang akan diverifikasi dengan setiap tanda tangan yang telah dilatih
            nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nbrs.fit(trained_features)
            distances, indices = nbrs.kneighbors(signature_features.reshape(1, -1))

            # Membandingkan jarak dengan ambang batas
            if distances[0][0] < threshold:
                return True, (1 - distances[0][0] / threshold) * 100  # Menghitung persentase keakuratan
            else:
                return False, (1 - distances[0][0] / threshold) * 0  # Menghitung persentase keakuratan

        # Inisialisasi daftar trained_signatures
        trained_signatures = []

        # Mendapatkan daftar file gambar tanda tangan dari folder data latih
        folder_path = "dataLatih"  # Ganti dengan path folder data latih Anda

        # Loop melalui setiap file dalam folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            signature_image = cv2.imread(file_path)

            # Ekstraksi fitur tanda tangan dari gambar
            signature_features = extract_signature_features(signature_image)

            # Tambahkan fitur tanda tangan ke dalam daftar trained_signatures
            trained_signatures.append(signature_features)
        # Menentukan ambang batas untuk verifikasi
        verification_threshold = 0.4

        capture = cv2.VideoCapture(1)  # Mengakses feed kamera dengan ID 0

        while True:
            ret, frame = capture.read()  # Membaca frame dari feed kamera

            cv2.imshow('verifikasi', frame)

            # Konversi ke citra ke skala abu-abu
            # grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Menggunakan metode adaptif untuk menghasilkan citra biner





            # # Konversi ke citra ke skala abu-abu
            # grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #
            # # Menggunakan metode adaptif untuk menghasilkan citra biner
            # binary = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            #
            # cv2.imshow('verifikasi', binary)

            # Keluar jika tombol 'q' ditekan
            if cv2.waitKey(1) == ord('q'):
                break

            if cv2.waitKey(1) == 13:  # Tekan tombol Enter untuk melakukan verifikasi
                signature_to_verify = frame  # Gunakan frame saat ini untuk verifikasi
                signature_features = extract_signature_features(signature_to_verify)

                # Melakukan verifikasi tanda tangan
                verification_result, accuracy = verify_signature(signature_features, trained_signatures, verification_threshold)

                if accuracy > 70:
                    print("Tanda tangan terverifikasi.")
                else:
                    print("Tanda tangan tidak terverifikasi.")

                # Menampilkan persentase keakuratan
                print("Accuracy: {:.2f}%".format(accuracy))

                self.label_3.setText(" {:.2f}%".format(accuracy))

        capture.release()
        cv2.destroyAllWindows()


    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:  # row[0],col[1],channel[2]
            if self.Image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
        elif windows == 2:
            self.label2.setPixmap(QPixmap.fromImage(img))
            self.label2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label2.setScaledContents(True)
        elif windows == 3:
            self.label3.setPixmap(QPixmap.fromImage(img))
            self.label3.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label3.setScaledContents(True)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Verifikasi tanda tangan')
window.show()
sys.exit(app.exec_())

