from PIL import Image, ImageOps
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
from numba import jit
import pickle
import time
from sklearn.cluster import MiniBatchKMeans
import os
from shutil import copyfile
import multiprocessing


def get_global_moments(f_list):
    n = 0
    s = 0
    for i in tqdm(f_list, desc="Calculating global mean"):
        image = Image.open(i, "r")
        image = ImageOps.grayscale(image)  # makes it greyscale
        img = np.asarray(image.getdata(), dtype=np.float64).reshape((image.size[1], image.size[0]))
        image.close()
        n += (img.shape[0]*img.shape[1])
        s += np.sum(img)

    mean = s / float(n)
    diff = 0
    for i in tqdm(f_list, desc="Calculating global std"):
        image = Image.open(i, "r")
        image = ImageOps.grayscale(image)  # makes it greyscale
        img = np.asarray(image.getdata(), dtype=np.float64).reshape((image.size[1], image.size[0]))
        image.close()
        diff += np.sum(np.square(img - mean))
    std = np.sqrt(diff / float(n - 1))

    return mean, std


@jit(nopython=True)
def rmse(a, b):
    rms = np.sqrt(np.mean(np.square(a-b)))
    return rms


@jit(nopython=True)
def psnr(a, b):
    r = np.max(np.max(a), np.max(b)) - np.min(np.min(a), np.min(b))
    mse = float(np.mean(np.square(a-b)))
    out = 10*np.log10(r**2/mse)
    return out


@jit(nopython=True)
def standardize(img, mean, std):
    img = (img-mean) / float(std)
    return img


def compare_images(a, f_list, mean, std, process_n, return_dict, metric):
    comp_list = list()
    for j in range(0, len(f_list)):
        f2 = f_list[j]
        image2 = Image.open(f2, "r")
        image2 = ImageOps.grayscale(image2)  # makes it greyscale
        b = np.asarray(image2.getdata(), dtype=np.float64).reshape((image2.size[1], image2.size[0]))
        image2.close()
        b = standardize(b, mean, std)
        if metric.lower() == "rmse":
            comp_list.append(rmse(a, b))
        elif metric.lower() == "psnr":
            comp_list.append(psnr(a, b))
        else:
            raise Exception("Did not recognize metric. Make sure that it is one of rmse, psnr, ssim, ms-ssim")
    return_dict[process_n] = comp_list


def multi_compare(a, i, f_list, similarity_matrix, mean, std, metric):
    # setup multiprocessing

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = list()

    n_processes = os.cpu_count()
    # stop running processes once there is less than 100 elements in the f_list...slower than single process
    if len(f_list) <= 100 or len(f_list) <= n_processes:
        for j in range(0, len(f_list)):
            f2 = f_list[j]
            image2 = Image.open(f2, "r")
            image2 = ImageOps.grayscale(image2)  # makes it greyscale
            b = np.asarray(image2.getdata(), dtype=np.float64).reshape((image2.size[1], image2.size[0]))
            image2.close()
            b = standardize(b, mean, std)
            if metric.lower() == "rmse":
                m = rmse(a, b)
            elif metric.lower() == "psnr":
                m = psnr(a, b)
            else:
                raise Exception("Did not recognize metric. Make sure that it is one of rmse, psnr, ssim, ms-ssim")
            similarity_matrix[i, j] = m
            similarity_matrix[j, i] = m
        return similarity_matrix

    chunks = np.array_split(f_list[i+2:], n_processes)

    for process_n, c in enumerate(chunks):
        p = multiprocessing.Process(target=compare_images, args=(a, c, mean, std, process_n, return_dict, metric))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    all_values = list()
    for j in return_dict.keys():
        all_values += return_dict[j]
    for j, val in enumerate(all_values):
        similarity_matrix[i, j] = val
        similarity_matrix[j, i] = val
    return similarity_matrix


def generate_similarity_matrix(f_list, mean, std, metric):

    similarity_matrix = np.zeros(shape=(len(f_list), len(f_list)))
    for i, f1 in enumerate(tqdm(f_list, desc="Generating similarity matrix")):

        image = Image.open(f1, "r")
        image = ImageOps.grayscale(image)  # makes it greyscale
        a = np.asarray(image.getdata(), dtype=np.float64).reshape((image.size[1], image.size[0]))
        image.close()
        a = standardize(a, mean, std)
        similarity_matrix = multi_compare(a, i, f_list, similarity_matrix, mean, std, metric)
    return similarity_matrix


def add_entry_to_similarity(f_list, new_img, similarity_matrix, mean, std):
    similarity_matrix = np.append(similarity_matrix, np.zeros(len(f_list)), axis=1)
    similarity_matrix = np.append(similarity_matrix, np.zeros(len(f_list)+1), axis=0)

    image = Image.open(new_img, "r")
    image = ImageOps.grayscale(image)  # makes it greyscale
    a = np.asarray(image.getdata(), dtype=np.float64).reshape((image.size[1], image.size[0]))
    a = standardize(a, mean, std)
    image.close()

    for i, f1 in enumerate(tqdm(f_list, desc="Generating similarity matrix")):
        image = Image.open(f1, "r")
        image = ImageOps.grayscale(image)  # makes it greyscale
        b = np.asarray(image.getdata(), dtype=np.float64).reshape((image.size[1], image.size[0]))
        image.close()
        b = standardize(b, mean, std)
        rms = rmse(a, b)
        similarity_matrix[i, len(f_list)] = rms
        similarity_matrix[len(f_list), i] = rms
    return similarity_matrix


def cluster_similarity_matrix():
    print("Loading similarity matrix")
    similarity_matrix = pd.read_csv("similarity_matrix.csv")

    model = MiniBatchKMeans(n_clusters=8, verbose=1).fit(similarity_matrix.values)

    labels = model.labels_
    count = 0
    for i, lab in enumerate(tqdm(labels)):
        if count == 1000:
            break
        count += 1
        img_path = similarity_matrix.columns[i]

        if not os.path.isdir("out/%s/" % lab):
            os.mkdir("out/%s/" % lab)
        copyfile(img_path, "out/%s/%s.jpg" % (lab, os.path.basename(img_path)))


    # Now place an image into each of the things


def main(calc_moments=False, ):
    f_list = glob.glob("/home/timmer/Desktop/etimmer-bioturbationdl-ac05fa22a1fa/deep_learning/*/*/*.jpg")
    # stack_images(f_list)
    if calc_moments is True:
        global_mean, global_std = get_global_moments(f_list)
        pickle.dump([global_mean, global_std], open("mean_std.pkl", "wb"))
    else:
        moments = pickle.load(open("mean_std.pkl", "rb"))
        global_mean, global_std = moments
    similarity_matrix = generate_similarity_matrix(f_list, global_mean, global_std, metric="rmse")
    pd.DataFrame(similarity_matrix).to_csv("test.csv")


if __name__ == "__main__":

    main()