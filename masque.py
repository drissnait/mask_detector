#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import os
# minimiser les messages du debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


# load our serialized model from disk
caffe_model = 'deploy.prototxt.txt'
caffe_trained = 'res10_300x300_ssd_iter_140000.caffemodel'
caffe_confidence = 0.30
model_folder = './model/'
mask_model = "mask_mobile_net.h5"

# Lire un modele réseaux depuis le modele caffe
net = cv2.dnn.readNetFromCaffe(model_folder + caffe_model,
                               model_folder + caffe_trained
                               )

print("[INFO] Chargement du modèle...")
model = load_model(model_folder + mask_model)
print("[INFO] Le modèle chargé avec succés.")

# Detecter les visages dans une image et appeler le prédicteur des masques
def detect_face_cnn(image):
    if image is not None:
        # Obtenir les dimensions de l'image
        (h, w) = image.shape[:2]

        # Redimensinoner l'image
        image_resized = cv2.resize(image, (300, 300))

        # Créer un blob 4 dimensionnel depuis l'image redimensionner
        blob = cv2.dnn.blobFromImage(image_resized,
                                     1.0,
                                     (300, 300),
                                     (104.0,
                                      177.0,
                                      123.0))
        # Mettre le blob comme un entré pour l'objet NET
        net.setInput(blob)

        # Executer un pass formward pour calculer le calque de sortie (les detections)
        detections = net.forward()

        # Pour chaque détection..
        for i in range(0, detections.shape[2]):
            # Extraire la confidence (probabilité) associé à la prédiction
            confidence = detections[0, 0, i, 2]

            # Filtrer les faibles détections en assurant que la "confidence" est
            # supérieur à la valeur minimum
            if confidence > caffe_confidence:
                # Calculer les (x, y)-coordoniates de la boîte englobante pour l'objet
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                try:
                    # Rogner la boîte englobante dans img_crop
                    img_crop = image[startY - 10:endY + 10, startX - 10:endX + 10]

                    # Prédire le port du masque
                    pred, pred_res = predict_mask(img_crop)

                    # Formatter la sortie (texte, couleur et pourcentage)
                    label = "masque" if pred_res == 0 else "pas de masque"
                    color = (0, 255, 0) if pred_res == 0 else (0, 0, 255)
                    label = '{} {:.2f}%'.format(label, round(confidence, 2))

                    # Afficher le pourcentage et le cadre sur le visage
                    cv2.putText(image, label, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
                except:
                    print("found crop errors {}".format(round(confidence, 2)))

        return image
    else:
        print("image not found!")


# Prédire si un visage porte un masque ou pas
def predict_mask(image):
    # Redimensionner l'image et la formater
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Faire la prédiction
    pred = model.predict(image)
    pred_res = pred.argmax(axis=1)[0]

    return pred, pred_res


### MAIN AREA
if __name__ == "__main__":
    # Obtenir la caméra (le flux vidéo d'index 0)
    cam = cv2.VideoCapture(0)

    while cam.isOpened():
        try:
            # Obtenir l'image depuis la cam
            ret, frame = cam.read()

            # Detecter les visages + port du masque
            frame = detect_face_cnn(frame)

            # Afficher la résultat
            cv2.imshow("Image", frame)

            # Cas d'arrêt (l'appuie sur la touche 'S' du clavier)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        except KeyboardInterrupt:
            # Libérer la caméra
            cam.release()
            break

    # Libérer la caméra
    cam.release()

    # Fermer la fenetre
    cv2.destroyAllWindows()
