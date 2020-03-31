from __future__ import print_function, division
import cv2
import numpy as np
import unittest


# class test(unittest.TestCase):
#     def setUp(self, imm, iml):
#         img_memory = imm
#         img_load = iml

#     def test_load(self, img_memory, img_load):
#         self.assertEquals(img_memory, img_load)

def downsample(img):
    print(img.shape)
    rows, cols = map(int, img.shape)
    img_dwn = cv2.pyrDown(img, dstsize=(cols//2, rows//2))
    return img_dwn

def down_sample_image(og_img):
    img_dwn_1 = downsample(og_img)
    img_dwn_2 = downsample(img_dwn_1)
    return img_dwn_2

def apply_filter(img):
    # -- Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(img,(5,5),1.5,1.5)
    
    # Bilateral Filtering & Resizing for visual aid
    bilat_fil_img = cv2.bilateralFilter(gaussian_blur, 1, 0.1, 5, 1)
    bilat_fil_resize = cv2.resize(bilat_fil_img, (bilat_fil_img.shape[1]*10,bilat_fil_img.shape[0]*10), interpolation=cv2.INTER_NEAREST)

    return bilat_fil_img

def compute_graph(bilat_fil_img):
    crop_vessel = bilat_fil_img.copy()
    crop_vessel_resize = cv2.resize(crop_vessel, dsize=(crop_vessel.shape[1]//2, crop_vessel.shape[0]//2), interpolation=cv2.INTER_NEAREST)
    
    # -- Define a neighborhood size for computing mean and variance around a pixel
    [mean_output, variance_output, pad] = mean_variance(crop_vessel_resize, 3)
    [mean_output_add_one, variance_output_add_one, pad_add_one] = mean_variance_add_one(crop_vessel_resize, 3)
    # [mean_output_add_one_test, variance_output_add_one_test, pad_add_one_test] = mean_variance_add_one(crop_vessel_test, 3)
    
    # -- Compute decreasing variance graph
    dec_var_graph = graph(variance_output)
    colored_img = color_by_graph(crop_vessel_resize, dec_var_graph)
    cv2.imwrite("colored_img.PNG", cv2.resize(colored_img, dsize=(crop_vessel_resize.shape[1]*5, crop_vessel_resize.shape[0]*5), interpolation=cv2.INTER_NEAREST))
    
    dec_var_graph_add_one = graph(variance_output_add_one)
    colored_img_add_one = color_by_graph(crop_vessel_resize, dec_var_graph_add_one)
    cv2.imwrite("colored_img_add_one.PNG", cv2.resize(colored_img_add_one, dsize=(crop_vessel_resize.shape[1]*5, crop_vessel_resize.shape[0]*5), interpolation=cv2.INTER_NEAREST))
    
    dec_var_graph_add_one_test = graph(variance_output_add_one)
    # colored_img_add_one_test = color_by_graph(crop_vessel_test, dec_var_graph_add_one_test)
    # cv2.imwrite("colored_img_add_one_test.PNG", cv2.resize(colored_img_add_one_test, dsize=(crop_vessel_test.shape[1]*5, crop_vessel_test.shape[0]*5), interpolation=cv2.INTER_NEAREST))

    # -- Highlight cluster centers
    cluster_centers, color_img_resize = render(dec_var_graph, colored_img)
    cluster_centers_add_one, color_img_resize_add_one = render(dec_var_graph_add_one, colored_img_add_one)
    # cluster_centers_add_one_test, color_img_resize_add_one_test = render(dec_var_graph_add_one_test, colored_img_add_one_test)


    # -- Saving relevant files
    # np.save("Cropped_vessel",crop_vessel)
    # cv2.imwrite("Bilateral_filtered.PNG", bilat_fil_img)
    # cv2.imwrite("Cropped_Vessel.PNG", cv2.resize(crop_vessel_resize, dsize=(crop_vessel_resize.shape[1]*5, crop_vessel_resize.shape[0]*5), interpolation=cv2.INTER_NEAREST))
    # cv2.imwrite("Hilighed_Cluster_Centers.PNG", color_img_resize)
    # cv2.imwrite("Hilighed_Cluster_Centers_add_one_test2.PNG", color_img_resize_add_one)
    # cv2.imwrite("Hilighed_Cluster_Centers_add_one_test.PNG", color_img_resize_add_one_test)
    # np.save("Mean_Intensity", mean_output)
    # np.save("Variance_Intensity", variance_output)
    # np.save("Descending_Variance_Graph", dec_var_graph)

    # return color_img_resize_add_one
    return cv2.resize(colored_img_add_one, dsize=(crop_vessel_resize.shape[1]*5, crop_vessel_resize.shape[0]*5), interpolation=cv2.INTER_NEAREST)


def mean_variance(img, radius):
    kernel = np.multiply(np.ones((radius*2 + 1,radius*2 + 1)), 1)
    iH, iW = img.shape[0], img.shape[1] # iH: Img Height, iW: Img Width
    kH, kW = kernel.shape[0], kernel.shape[1] # kH: kernel Height. kW: kernel weigth

    pad = (kW-1)//2
    img_pad = cv2.copyMakeBorder(img.astype(np.int16), pad, pad, pad, pad, cv2.BORDER_CONSTANT, -1) # padded image
    
    
    # Define np arrays
    img_mean = np.zeros((iH, iW), dtype = "float64")
    img_variance = np.zeros((iH, iW), dtype = "float64")
    idx = 0
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = img_pad[y-pad:y+pad+1, x-pad:x+pad+1] # defining roi (region of interest)
            denominator = np.count_nonzero(roi + 1) # keeping count of nonzero elements
            if idx == 0:
                print("denominator", denominator)
                idx += 1
            mean = (roi*kernel).sum()/denominator # finding mean of nonzero pixels
            intermed = np.subtract(roi,mean)
            variance = (intermed*intermed).sum()/denominator
            img_mean[y-pad, x-pad] = mean
            img_variance[y-pad, x-pad] = variance

    mean_output = (img_mean).astype("uint8")
    variance_output = (img_variance / np.max(img_variance)).astype("float32")

    return mean_output, variance_output, pad

def mean_variance_add_one(img, radius):
    kernel = np.multiply(np.ones((radius*2 + 1,radius*2 + 1)), 1)
    iH, iW = img.shape[0], img.shape[1] # iH: Img Height, iW: Img Width
    kH, kW = kernel.shape[0], kernel.shape[1] # kH: kernel Height. kW: kernel weigth

    pad = (kW-1)//2
    img_add_one = img + 1
    # img_pad = cv2.copyMakeBorder(img.astype(np.int16), pad, pad, pad, pad, cv2.BORDER_CONSTANT, -1) # padded image
    img_pad_add_one = cv2.copyMakeBorder(img_add_one.astype(np.int16), pad, pad, pad, pad, cv2.BORDER_CONSTANT, -1) # padded image
    
    
    # Define np arrays
    img_mean = np.zeros((iH, iW), dtype = "float64")
    img_variance = np.zeros((iH, iW), dtype = "float64")

    idy = 0
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # roi = img_pad[y-pad:y+pad+1, x-pad:x+pad+1] # defining roi (region of interest)
            roi_add_one = img_pad_add_one[y-pad:y+pad+1, x-pad:x+pad+1]
            # print("roi\n", roi)
            # print("roi + 1\n", roi_adding_one)
            # denominator = np.count_nonzero(roi + 1) # keeping count of nonzero elements
            denominator_new = np.count_nonzero(roi_add_one)
            # print("denominator",denominator)
            if idy == 0:
                print("denominator_new",denominator_new)
                idy += 1
            # roi[roi==-1] = 0
            mean = (roi_add_one*kernel).sum()/denominator_new # finding mean of nonzero pixels
            intermed = np.subtract(roi_add_one,mean)
            variance = (intermed*intermed).sum()/denominator_new
            img_mean[y-pad, x-pad] = mean
            img_variance[y-pad, x-pad] = variance

    mean_output = (img_mean).astype("uint8")
    variance_output = (img_variance / np.max(img_variance)).astype("float32")

    return mean_output, variance_output, pad


def graph_old(img_variance, pad):
    img_pad = cv2.copyMakeBorder(img_variance, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    descending_var = np.zeros((img_variance.shape), dtype=object)
    
    for y in np.arange(pad,img_pad.shape[0]-pad):
        for x in np.arange(pad, img_pad.shape[1]-pad):
            neighbors = np.array([img_pad[y][x], img_pad[y-1][x], img_pad[y][x+1], img_pad[y+1][x], img_pad[y][x-1]])
            switcher = {0:(y-pad,x-pad), 1:(y-pad-1,x-pad), 2:(y-pad,x), 3:(y,x-pad), 4:(y-pad,x-pad-1)}
            min = np.argmin(neighbors)
            try:
                assert(switcher.get(min)[0]<img_variance.shape[0] and switcher.get(min)[1]<img_variance.shape[1])
            except:
                print("OOPS!", switcher.get(min), img_variance.shape)
                
            descending_var[y-pad][x-pad] = switcher.get(min)
    return descending_var



def graph(img_variance):
    img_pad = cv2.copyMakeBorder(img_variance, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    descending_var = np.zeros((img_variance.shape), dtype=object)
    centers = 0
    for y in range(img_variance.shape[0]):
        for x in range(img_variance.shape[1]):
            neighbors = [(y    , x    ),
                         (y - 1, x    ),
                         (y + 1, x    ),
                         (y    , x - 1),
                         (y    , x + 1)]
            values = np.array([img_pad[a+1,b+1] for (a,b) in neighbors])
            min = np.argmin(values)
            if min == 0:
                centers = centers + 1
            try:
                assert(neighbors[min][0]<img_variance.shape[0] and neighbors[min][1]<img_variance.shape[1])
            except:
                print("Neighbors out of range!", neighbors[min], img_variance.shape)
                
            descending_var[y][x] = neighbors[min]
    # print(centers)
    return descending_var


def render_old(graph, crop_vessel_resize):
    cluster_centers = []
    rows, cols = graph.shape

    color_img = cv2.cvtColor(crop_vessel_resize,cv2.COLOR_GRAY2BGR)
    count = 0
    for i in range(0,rows):
        for j in range(0,cols):
            if graph[i][j] == (i,j):
                color_img[i][j] = (0, 255, 255)
                cluster_centers.append((i,j))
                count+=1
    color_img_resize = cv2.resize(color_img, dsize=(color_img.shape[1]*5, color_img.shape[0]*5), interpolation=cv2.INTER_NEAREST)
       
    return cluster_centers, color_img_resize

def render(graph, img):

    cluster_centers = []
    rows, cols = graph.shape    
    scaling = 5
    color_img = cv2.resize(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR), (cols * scaling, rows * scaling), interpolation=cv2.INTER_NEAREST)
    count = 0
    for i in range(0,rows):
        for j in range(0,cols):
            if graph[i][j] == (i,j):
                color_img[i*scaling + scaling//2][j*scaling + scaling//2] = (0, 255, 255)
                cluster_centers.append((i,j))
                count+=1

    print(count)
    return cluster_centers, color_img


def color_by_graph(img, graph):
    new_img = img.copy()
    visited = np.ones(img.shape) * -1
    rows, cols = graph.shape
    for i in range(0,rows):
        for j in range(0,cols):
            color_by_graph_helper(new_img, graph, visited, (i,j), 0)
    return new_img

def color_by_graph_helper(img, graph, visited, idx, depth):
    i,j = idx
    # print(idx, graph[i,j], depth)
    if depth > 100:
        print("TOO DEEP")
        quit()
    # print(depth, (i,j), graph[i,j], img.shape)
    if visited[i,j] != -1:
        img[i,j] = visited[i,j]
        return visited[i,j]
    if graph[i,j] != idx:
        visited[i,j] = color_by_graph_helper(img, graph, visited, graph[i,j], depth+1)
        img[i,j] = visited[i,j]
        return visited[i,j]
    else:
        # print("RETURNING")
        visited[i,j] = img[i,j]
        return visited[i,j]


def test(img, img1):
    print(img.shape)
    print(img1.shape)
    cv2.imshow("Original Image",img)
    cv2.imshow("New Image",img1)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--path1', type=str, help='path to image 1')
    parse.add_argument('--path2', type=str, help='path to image 2')
    args = parse.parse_args()
    # -- Read Image
    og_img = cv2.imread(args.path1,0)
    # og_img = cv2.imread("downloaded_vessel.jpg",0)
    test_img = cv2.imread(args.path2,0)
    print(test_img.shape)
    # -- Downsample by factor of 4 [by factor of 2 in each following step]
    img_dwn_1 = downsample(og_img)
    img_dwn_2 = downsample(img_dwn_1)
    print(img_dwn_2.shape)
    
    # cv2.imshow("test", test_img)
    # cv2.waitKey(0)
    # -- Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(img_dwn_2,(5,5),1.5,1.5)
    
    # Bilateral Filtering & Resizing for visual aid
    bilat_fil_img = cv2.bilateralFilter(gaussian_blur, 1, 0.1, 5, 1)
    bilat_fil_resize = cv2.resize(bilat_fil_img, (bilat_fil_img.shape[1]*10,bilat_fil_img.shape[0]*10), interpolation=cv2.INTER_NEAREST)
    
    # -- Manually isolating area of interest & Resizing it for visual aid
    # crop_vessel = bilat_fil_resize[250:520, 1155:1595]
    crop_vessel = bilat_fil_img.copy()
    crop_vessel_resize = cv2.resize(crop_vessel, dsize=(crop_vessel.shape[1]//2, crop_vessel.shape[0]//2), interpolation=cv2.INTER_NEAREST)
    crop_vessel_test = cv2.resize(test_img, dsize=(test_img.shape[1]//6, test_img.shape[0]//6), interpolation = cv2.INTER_NEAREST)
    print(crop_vessel_resize.shape)
    print(crop_vessel_test.shape)
    # -- Define a neighborhood size for computing mean and variance around a pixel
    [mean_output, variance_output, pad] = mean_variance(crop_vessel_resize, 3)
    [mean_output_add_one, variance_output_add_one, pad_add_one] = mean_variance_add_one(crop_vessel_resize, 3)
    [mean_output_add_one_test, variance_output_add_one_test, pad_add_one_test] = mean_variance_add_one(crop_vessel_test, 3)
    
    # -- Compute decreasing variance graph
    dec_var_graph = graph(variance_output)
    colored_img = color_by_graph(crop_vessel_resize, dec_var_graph)
    cv2.imwrite("colored_img.PNG", cv2.resize(colored_img, dsize=(crop_vessel_resize.shape[1]*5, crop_vessel_resize.shape[0]*5), interpolation=cv2.INTER_NEAREST))
    
    dec_var_graph_add_one = graph(variance_output_add_one)
    colored_img_add_one = color_by_graph(crop_vessel_resize, dec_var_graph_add_one)
    cv2.imwrite("colored_img_add_one.PNG", cv2.resize(colored_img_add_one, dsize=(crop_vessel_resize.shape[1]*5, crop_vessel_resize.shape[0]*5), interpolation=cv2.INTER_NEAREST))
    
    dec_var_graph_add_one_test = graph(variance_output_add_one)
    colored_img_add_one_test = color_by_graph(crop_vessel_test, dec_var_graph_add_one_test)
    cv2.imwrite("colored_img_add_one_test.PNG", cv2.resize(colored_img_add_one_test, dsize=(crop_vessel_test.shape[1]*5, crop_vessel_test.shape[0]*5), interpolation=cv2.INTER_NEAREST))

    test(colored_img, colored_img_add_one)
    
    # -- Highlight cluster centers
    cluster_centers, color_img_resize = render(dec_var_graph, colored_img)
    cluster_centers_add_one, color_img_resize_add_one = render(dec_var_graph_add_one, colored_img_add_one)
    cluster_centers_add_one_test, color_img_resize_add_one_test = render(dec_var_graph_add_one_test, colored_img_add_one_test)


    # -- Saving relevant files
    np.save("Cropped_vessel",crop_vessel)
    cv2.imwrite("Bilateral_filtered.PNG", bilat_fil_resize)
    cv2.imwrite("Cropped_Vessel.PNG", cv2.resize(crop_vessel_resize, dsize=(crop_vessel_resize.shape[1]*5, crop_vessel_resize.shape[0]*5), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite("Hilighed_Cluster_Centers.PNG", color_img_resize)
    cv2.imwrite("Hilighed_Cluster_Centers_add_one_test2.PNG", color_img_resize_add_one)
    cv2.imwrite("Hilighed_Cluster_Centers_add_one_test.PNG", color_img_resize_add_one_test)
    np.save("Mean_Intensity", mean_output)
    np.save("Variance_Intensity", variance_output)
    np.save("Descending_Variance_Graph", dec_var_graph)