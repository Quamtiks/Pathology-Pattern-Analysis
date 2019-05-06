
import cv2
import numpy as np
import random

'''Funcion encargada de buscar circulos de un determinado rango de
radio en la imagen, devuelve una lista compuesta de circulos indican
la ubicacion en donde se encontro un circulo aceptable
para el rango de busqueda'''
def get_circulos_hough(imagen, minRadio, maxRadio):
    
    img = cv2.imread(imagen)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detecta circulos en la imagen
    circulos = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT,  1.2, 50, param1=18, param2=31, minRadius = minRadio, maxRadius = maxRadio)
    return circulos


'''Funcion encargada de dibujar en la imagen original
todos los circulos encontrados por medio del metodo de
houghs'''              
def dibujar_circulos(circulos, color, img):
    salida = img
    # Asegura de dibujar si la lista no es nula
    if circulos is not None:
            # Transforma las coordenadas (x, y) y el radio a enteros
            circulos = np.round(circulos[0, :]).astype("int")
            
            for (x, y, r) in circulos:
                    # Dibuja un circulo en la ubicacion correspondiente
                    # y en su centro un rectangulo
                    cv2.circle(salida, (x, y), r, color, 4)
                    cv2.rectangle(salida, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
     
            # Muestra la salida de la imagen
            cv2.imshow("out", salida)

'''Funcion encargada de trabajar en la imagen original
de tal manera que sea mas sencillo extraer sus
datos para posteriores funciones y trabajamiento
del programa'''             
def pre_procesamieto(imagen):
    img=cv2.imread(imagen)
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    '''Creacion de la mascara que extraer todos las tonalidades
        de color rojo encontradas que seran aquellas celulas sanas'''
    # mascara de rojo minimo (0-10) 
    min_color = np.array([0,50,50])
    max_color = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, min_color, max_color)
    # mascara de rojo maximo (170-180)
    min_color = np.array([170,50,50])
    max_color = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, min_color, max_color)
    # union de mascaras
    mask = mask0+mask1
    ret, mask_inv= cv2.threshold(mask,180,255,cv2.THRESH_BINARY_INV)

    # pone en zero cualquier posiciÃ³n distinta de la mascara
    salida_img = img.copy()
    salida_img[np.where(mask==0)] = 0

    '''Creacion de la mascara que extraer todos las tonalidades
        de color verde encontradas que seran aquellas celulas enfermas'''
    # mascara de verde minimo(0-10)
    min_color = np.array([25,50,50])
    max_color = np.array([40,255,255])
    mask0 = cv2.inRange(img_hsv, min_color, max_color)
    # mascara de verde maximo (170-180)
    min_color = np.array([60,50,50])
    max_color = np.array([85,255,255])
    mask1 = cv2.inRange(img_hsv, min_color, max_color)
    # union de mascaras
    mask = mask0+mask1

    
    salida_img_inv = img.copy()
    salida_img_inv[np.where(mask==0)] = 0

    '''Crea las iamgenes en la ruta donde se corre el codigo con el
        fin de poder anilizarlas mas facilmente'''
    cv2.imwrite("sanas.jpg", salida_img)
    kernel = np.ones((3,3),np.uint8)
    salida_img_inv = cv2.dilate(salida_img_inv,kernel,iterations = 15)
    salida_img_inv = cv2.erode(salida_img_inv,kernel,iterations = 11)
    cv2.imwrite("enfermas.jpg", salida_img_inv)


if __name__ == "__main__":

    salida = cv2.imread("influenza_ifa.jpg")
    pre_procesamieto("influenza_ifa.jpg")
    
    circulos = get_circulos_hough("sanas.jpg", 19, 36)
    sanas = len(circulos[0])
    dibujar_circulos(circulos, (0,255,0), salida)
    
    circulos = get_circulos_hough("enfermas.jpg", 18, 50)
    print("Se encontraron {} celulas enfermas".format(len(circulos[0])))
    print("Se encontraron {} celulas sanas".format(sanas-len(circulos[0])))
    print("Hay un porcentaje {} % de celulas en riesgo".format( (len(circulos[0])/(sanas+len(circulos[0])))*100 ))
    dibujar_circulos(circulos, (255,0,0), salida)
