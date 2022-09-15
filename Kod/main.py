import cv2
import numpy as np
import keyboard
import sqlite3

cap = cv2.VideoCapture(2)

baglanti = sqlite3.connect("database.db")#database oluşturur, yoksa kurar, varsa bağlanır
imlec = baglanti.cursor()

imlec.execute("CREATE TABLE IF NOT EXISTS ornekler (ornek1 INT)")

def eslestirme():
    gelen = cv2.imread('ornek.jpg')
    image_roi, rounds = processing(gelen, 30)
    image_roi = recflection_remove(image_roi)
    image_nor = daugman_normalizaiton(image_roi, 60, 360, rounds, 55)
    cv2.imwrite("daugman.jpg", image_nor)
    özellik = özellikcikar(image_nor)
    print(özellik)
    deger = karşılaştır(özellik)
    return deger

def processing(image_path, r):
    image_path = cv2.resize(image_path, (640, 480), interpolation=cv2.INTER_LINEAR) #boyutu 640x480 yapar
    image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY) #graya dönüştürür
    gray = cv2.medianBlur(image_path, 11) #Blur yapar ve gürültüleri siler
    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Thresh binnary eşik değeri ile siyah ve beyazı ayırır
    cv2.imwrite('gray.jpg', gray)
    cv2.imwrite('processing.jpg', image_path)#image_path dosyasını processing.jpg olarak kaydeder
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=ret, param2=30, minRadius=20,
                               maxRadius=100) #Çemberleri bulur
    circles = circles[0, :, :]
    circles = np.int16(np.around(circles)) #Integer (-32768 to 32767) yuvarlama yapar
    for i in circles[:]: #Görüntüyü düzenler
        image_path = image_path[i[1] - i[2] - r:i[1] + i[2] + r, i[0] - i[2] - r:i[0] + i[2] + r]
        radus = i[2]/(2.5)
        print("radus:", radus)
    cv2.imwrite('circle.jpg', image_path)
    return image_path, radus

def recflection_remove(img):
    ret, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)#beyaz kısımları arttırır
    dst = cv2.inpaint(img, dilation, 4, cv2.INPAINT_TELEA)#gereksiz şeyleri siliyor
    cv2.imwrite('dilation.jpg', dilation)
    cv2.imwrite('dst.jpg', dst)
    return dst

def daugman_normalizaiton(image, height, width, r_in, r_out):# Daugman 640*480,width*height
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta değerleri (0, 2*pi, 2*pi/640) 0 ile 360 arasında 360/640 derece aralıklarla
    r_out = r_in + r_out

    flat = np.zeros((height, width, 3), np.uint8)#boş bir fotoğraf yarat
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # theta koordinat değerleri
            r_pro = j / height  # yarıçap koordinat(normalized) değerleri

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = image[int(Xc)][int(Yc)]  # pixellerin renkleri

            flat[j][i] = color
    return flat

def özellikcikar(image):
    u = 0
    for i in range(60):
        for j in range(360):
            r, g, b= image[i, j]
            u += r
            ortalama = int(u/(21600/1.3))
    ret, threshold = cv2.threshold(image, ortalama, 255, cv2.THRESH_BINARY_INV)

    print("ortalama:", ortalama)
    x = [0] * 21600
    l = 0
    for c in range(60):
        for d in range(360):
            r, g, b = threshold[c, d]
            if r == 255:
                x[l] = 1
            else:
                x[l] = 0
            l = l + 1
    l = 0
    cv2.imwrite('the.jpg', threshold)
    return x

def karşılaştır(x):
    y = [0] * 21600
    sayac = 0
    imlec.execute("SELECT * FROM ornekler")
    data = imlec.fetchall()
    for i in range(21600):
        print(data[i])
        print(x[i])
        if x[i] == 0:
            y[i]=(0,)
        else: y[i] = (1,)
        print(y[i])
        if data[i] == y[i] :
                sayac += 1
    print("Benzerlik:", sayac)
    sonuc = sayac/21600
    return sonuc

a = 1
while True:
    _, frame = cap.read()
    frame2=frame
    frame2 = cv2.line(frame2, (0, 240), (125, 240), (255, 0, 0), 3)
    frame2 = cv2.line(frame2, (575, 240), (640, 240), (255, 0, 0), 3)
    cv2.imshow("resim", frame2)

    if a == 1:
        print("Lütfen eşleştirmeyi yapmak için 0'a basın!\n"
              "Kapatmak için herhangi bir tuşa basın!")
    if keyboard.is_pressed('0'):
        cv2.imwrite('ornek.jpg', frame)
        oran = eslestirme()
        print("Eşleştirme oranı:", float(oran))
        a = 1
        break
    else:
        a = 0
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
