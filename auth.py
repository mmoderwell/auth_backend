import sys
import face
import os.path
from os import stat
from os import listdir
import time
from scipy import misc
import json


def main():

    face_recognition = face.Recognition()
    face_tracking = face.Detection()

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, "uploads")

    json_file = os.path.join(path, "output.json")

    with open(json_file) as json_data:
        json_decoded = json.load(json_data)

    while True:
        #wait a half a second for each loop      
        time.sleep(.5)
        #get the file names of all frames in /uploads
        files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]

        for img in files:
            if (img != "output.json" and img not in json_decoded):
                img_path = os.path.join(path, img)
                # check if it was an accidental frame (O bytes)
                if os.stat(img_path).st_size != 0:
                    # Grab a frame from filepath and preprocess
                    frame = misc.imread(img_path)
                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:,:,0:3]

                    faces = face_tracking.find_faces(frame)

                    if len(faces) == 1:         
                        the_face = face_recognition.identify_name(faces)
                        print (img, the_face[0].name)
                        json_decoded[img] = the_face[0].name
                    else:
                        json_decoded[img] = 'None'

                    with open(json_file, 'w') as json_out:
                        json.dump(json_decoded, json_out)
                    

                





if __name__ == '__main__':
    main()
