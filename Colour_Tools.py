import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import cv2
from sklearn.cluster import KMeans

example_img = 'aquariusW.jpg'

def get_hexcolour(rgb):
    hex_color = '#{:02X}{:02X}{:02X}'.format(rgb[0], rgb[1], rgb[2])
    return hex_color

class Colour_Tool:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        dim = self.img.shape

        red_list = []
        green_list = []
        blue_list = []

        h, w = int(dim[0]/2), int(dim[1]/2)

        cv2.namedWindow('Window',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Window',w,h)
        cv2.imshow('Window', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        for pixel_row in self.img:
            # openCV2 works in BGR
            for blue, green, red in pixel_row:
                red_list.append(red)
                green_list.append(green)
                blue_list.append(blue)

        self.img_df = pd.DataFrame({'Red':red_list, 'Green':green_list, 'Blue':blue_list})

    def get_colour_clusters(self,no_of_clusters=5):

        model = KMeans(n_clusters=no_of_clusters)
        model.fit(self.img.reshape(-1,3))
        # need to reorder to R,G,B from B,G,R

        updated_clusters = []
        for cluster in model.cluster_centers_:
            bgr_list = list(cluster)
            rgb_list = bgr_list[::-1]
            updated_clusters.append(rgb_list)
        cluster_array = np.array(updated_clusters)
        cluster_array.reshape(-1,1)

        colour_palette = np.uint8(cluster_array)

        hex_list = []
        for rgb in colour_palette:
            hex = get_hexcolour(rgb)
            print(hex)
            hex_list.append(hex)

        plt.figure(figsize=(no_of_clusters+2, 3))
        plt.imshow([colour_palette])
        plt.title(hex_list)
        plt.axis('off')
        plt.show()

    def colour_spectrum(self,start_rgb=[25,0,0],gradient=10, rates=[10,10,10]):
        # start_rgb = [R, G, B]
        # rates = [Rx,Gx,Bx]
        gradient_list = []

        rate_r = rates[0]
        rate_g = rates[1]
        rate_b = rates[2]

        for n in range(gradient):
            r = int(start_rgb[0] + n * rate_r)
            g = int(start_rgb[1] + n * rate_g)
            b = int(start_rgb[2] + n * rate_b)

            gradient_list.append([r,g,b])

        colour = np.array(gradient_list).reshape(-1,3)
        colour = np.clip(colour,0,255)

        colour = np.uint8(colour)
        hex_list = []
        for col in colour:
            hex = get_hexcolour(col)
            hex_list.append(hex)

        plt.figure(figsize=(gradient + 2, 3))
        plt.title(hex_list)
        plt.imshow([colour])
        plt.axis('off')
        plt.show()


rates = [14,6,4]
t = Colour_Tool(example_img)
t.get_colour_clusters(no_of_clusters=5)
t.colour_spectrum(start_rgb=[15,158,213],gradient=12,rates=rates)
t.colour_spectrum(start_rgb=[0,0,0],gradient=12,rates=rates)
