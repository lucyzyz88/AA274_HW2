import argparse
import numpy as np
import tensorflow as tf
from utils import *

GRAPH = 'retrained_graph.pb'
with open('labels.txt', 'r') as f:
    LABELS = f.read().split()

def compute_brute_force_classification(image_path, nH=8, nW=8):
    raw_image = decode_jpeg(image_path)    # H x W x 3 numpy array (3 for each RGB color channel)

    #### EDIT THIS CODE
    print(np.shape(raw_image))
    Image_Height = np.shape(raw_image)[0]
    Image_Width = np.shape(raw_image)[1]
    #Set the window Size
    Window_Width = 150
    Window_Height = 90
    nH = Image_Height / Window_Height
    nW = Image_Width / Window_Width
    Height_Gap = (Image_Height-Window_Height)/(nH-1)
    Width_Gap = (Image_Width-Window_Width)/(nW-1)
    #window = np.zeros(nH,nw,Window_Height,Window_Width,3)
    window_predictions = np.zeros((nH,nW,3))

    for i in range(nH):
        for j in range(nW):
            #The i,j for the window is the ith,jth chunk 
            with tf.Session() as sess:
                window = raw_image[i*Height_Gap:(i*Height_Gap+Window_Height),j*Width_Gap:(j*Width_Gap+Window_Width),:]
                window_predictions[i][j] = classify_image(window, sess)


    # with tf.Session() as sess:
    #     window = raw_image   # the "window" here is the whole image
    #     window_prediction = classify_image(window, sess)

    # setting each window prediction to be the prediction for the whole image (just for something to plot)
    # do not turn this code in and claim that "your window padding is infinite"
    # window_predictions = np.array([[window_prediction[c,r,] for c in range(nW)] for r in range(nH)])

    ####

    return np.squeeze(np.array(window_predictions))

def compute_convolutional_KxK_classification(image):
    graph = tf.get_default_graph()
    classification_input_tensor =  graph.get_tensor_by_name('bottleneck_input/BottleneckInputPlaceholder:0')
    classification_output_tensor = graph.get_tensor_by_name('final_result:0')
    convolution_ouput_tensor = graph.get_tensor_by_name('bottleneck_grid:0')    # 'mixed_10/join:0' in Inception-v3
    K = convolution_ouput_tensor.shape[0]

    with tf.Session() as sess:
        convolution_output = run_with_image_input(convolution_ouput_tensor, image, sess)
        predictionsKxK = sess.run(classification_output_tensor,
                                  {classification_input_tensor: np.reshape(convolution_output, [K*K, -1])})
        return np.reshape(predictionsKxK, [K,K,-1])

def compute_and_plot_saliency(image):
    graph = tf.get_default_graph()
    logits_tensor = graph.get_tensor_by_name('final_training_ops/Wx_plus_b/logits:0')

    with tf.Session() as sess:
        logits = np.squeeze(run_with_image_input(logits_tensor, image, sess))
        top_class = np.argmax(logits)
        w_ijc = gradient_of_class_score_with_respect_to_input_image(image, top_class, sess)    # defined in utils.py

    #### EDIT THIS CODE
    M = np.maximum(np.abs(w_ijc), axis=2)
    #How big w_ijc the gradient matrix is
    i_range = np.shape(w_ijc)[0]
    j_range = np.shape(w_ijc)[1]
    M = np.zeros((i_range,j_range))
    #Loop through and get the biggest for w
    for i in range(i_range):
        for j in range(j_range):
            M_ij = np.max(np.absolute(w_ijc[i,j,:]))
            M[i][j] = M_ij
    print 'saliency',np.shape(w_ijc)
    ####

    plt.subplot(2,1,1)
    plt.imshow(M)
    plt.title('Saliency with respect to predicted class %s' % LABELS[top_class])
    plt.subplot(2,1,2)
    plt.imshow(decode_jpeg(image))
    plt.show()

def plot_classification(image_path, classification_array):
    nH, nW, _ = classification_array.shape
    image_data = decode_jpeg(image_path)
    aspect_ratio = float(image_data.shape[0]) / image_data.shape[1]
    plt.figure(figsize=(8, 8*aspect_ratio))
    p1 = plt.subplot(2,2,1)
    plt.imshow(classification_array[:,:,0], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[0])
    p1.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    p2 = plt.subplot(2,2,2)
    plt.imshow(classification_array[:,:,1], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[1])
    p2.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    p2 = plt.subplot(2,2,3)
    plt.imshow(classification_array[:,:,2], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[2])
    p2.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(image_data)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--scheme', type=str)
    FLAGS, _ = parser.parse_known_args()

    load_graph(GRAPH)
    if FLAGS.scheme == 'brute':
        plot_classification(FLAGS.image, compute_brute_force_classification(FLAGS.image, 8, 8))
    elif FLAGS.scheme == 'conv':
        plot_classification(FLAGS.image, compute_convolutional_KxK_classification(FLAGS.image))
    elif FLAGS.scheme == 'saliency':
        compute_and_plot_saliency(FLAGS.image)
    else:
        print 'Unrecognized scheme:', FLAGS.scheme