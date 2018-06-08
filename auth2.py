from scipy import misc
import sys

import cv2
import numpy as np
import tensorflow as tf
import os, time, random

import facenet
import face
import detect_face


FACE_DIR = "../train/"



def main():

    print("Starting image alignment.")
    #input directory
    dataset = facenet.get_dataset(FACE_DIR)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    print('Networks created. Using TensorFlow backend.')

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    #alignment config
    #Shuffles the order of images to enable alignment using multiple processes.
    random_order = True
    detect_multiple_faces = False
    #Image size (height, width) in pixels.
    image_size = 182
    #Margin for the crop around the bounding box (height, width) in pixels.
    margin = 44
    output_dir = "../processed/"

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    #bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    nrof_images_total = 0
    nrof_successfully_aligned = 0

    #for cls in dataset:
    output_class_dir = os.path.join(output_dir, dataset[0].name)

    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
        if random_order:
            random.shuffle(dataset[0].image_paths)

    for image_path in dataset[0].image_paths:
        nrof_images_total += 1
        filename = os.path.splitext(os.path.split(image_path)[1])[0]
        output_filename = os.path.join(output_class_dir, filename+'.png')
        print(image_path)
        if not os.path.exists(output_filename):
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim<2:
                    print('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:,:,0:3]

                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces>0:
                    det = bounding_boxes[:,0:4]
                    det_arr = []
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces>1:
                        if detect_multiple_faces:
                            for i in range(nrof_faces):
                                det_arr.append(np.squeeze(det[i]))
                        else:
                            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                            img_center = img_size / 2
                            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            det_arr.append(det[index,:])
                    else:
                        det_arr.append(np.squeeze(det))

                    for i, det in enumerate(det_arr):
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-margin/2, 0)
                        bb[1] = np.maximum(det[1]-margin/2, 0)
                        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                        nrof_successfully_aligned += 1
                        filename_base, file_extension = os.path.splitext(output_filename)
                        if detect_multiple_faces:
                            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                        else:
                            output_filename_n = "{}{}".format(filename_base, file_extension)
                        misc.imsave(output_filename_n, scaled)
                else:
                    print('Unable to align "%s"' % image_path)
                            
    #print('Total number of images: %d' % nrof_images_total)
    #print('Number of image successfully aligned: %d' % nrof_successfully_aligned)
    print ('Alignment complete.');



if __name__ == '__main__':
    main()