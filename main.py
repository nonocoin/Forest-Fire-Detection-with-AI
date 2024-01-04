import firebase_admin
from firebase_admin import credentials, storage
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2 as cv
global yangin_degil 
global yangin 
global olasilik

"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] ="2"
import tensorflow as tf

cred = credentials.Certificate("magnetar-fima-firebase-adminsdk-f9a8a-a5e7b0eb67.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'magnetar-fima.appspot.com'
})

percentage = None
bucket = storage.bucket()

file_path = "report1.jpg" 

blob = bucket.blob(file_path)

file_contents = blob.download_as_bytes()
    

with open("report1.jpg", "wb") as f:
    f.write(file_contents)
    
print("Görsel başarıyla indirildi.")
"""
best_model = tf.keras.models.load_model('final_model.h5')

def load_and_prep_image(filename, img_shape=300):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    return img


sample_img = 'report1.jpg'
class_names = ['Not-fire', 'Fire']


img = load_and_prep_image(sample_img)
pred = best_model.predict(tf.expand_dims(img, axis=0))
    
probability = pred[0][0]

percentage = probability * 100
olasilik = percentage
#print(f'Yangin Olma Olasiligi: {percentage:.2f}%')

if len(pred[0]) > 1:
    pred_class = class_names[pred.argmax()]
    yangin = 1
else:
    pred_class = class_names[int(tf.round(pred)[0][0])]
    yangin_degil = 1

file_to_upload = "file_to_upload.jpg"

if olasilik==0.5:
    yazi = str(f'{percentage:.2f}') + str("  Yangin var")

if olasilik>0.5:
    yazi = str(f'{percentage:.2f}') + str("  Yangin var")

if olasilik<0.5:
    yazi = str(f'{percentage:.2f}') + str("  Yangin Yok")

destination_blob_name = yazi  

#blob = bucket.blob(destination_blob_name)
#blob.upload_from_filename(file_to_upload)
scale_percent = 170 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

print(destination_blob_name)
resim = cv.imread(sample_img)
resized = cv.resize(resim, dim, interpolation = cv.INTER_AREA)
cv.imshow("resimler",resized)
cv.waitKey(0)

