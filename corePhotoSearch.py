from numba import jit
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import multiprocessing
import os
import shutil
import pickle


@jit(nopython=True)
def rmse(a, b):
    rms = np.sqrt(np.nanmean(np.square(a-b)))
    return rms


@jit(nopython=True)
def psnr(a, b):
    # A lot of silly variable declarations to appease numba...
    a_max = np.nanmax(a)
    b_max = np.nanmax(b)
    a_min = np.nanmin(a)
    b_min = np.nanmin(b)
    if a_max > b_max:
        max_r = a_max
    else:
        max_r = b_max
    if a_min < b_min:
        min_r = a_min
    else:
        min_r = b_min
    r = max_r - min_r
    mse = float(np.nanmean(np.square(a-b)))
    if mse == 0:
        return np.NaN
    out = 10*np.log10(r**2/mse)
    return out


@jit(nopython=True)
def hist(a, b):
    h_a, _ = np.histogram(a.flatten())
    h_b, _ = np.histogram(b.flatten())
    h = np.mean(np.abs(h_a - h_b))
    return h


@jit(nopython=True)
def compare(a, b, do_psnr, do_rmse, do_hist):
    out = [0.]
    if do_rmse is True:
        r = rmse(a, b)
        out.append(r)
    if do_psnr is True:
        p = psnr(a, b)
        out.append(p)
    if do_hist is True:
        h = hist(a, b)
        out.append(h)

    out.pop(0)

    return out


@jit(nopython=True)
def compare_img_tiles(img, n_tiles, overlap, do_psnr, do_rmse, do_hist):

    x_win = int(img.shape[1]/float(n_tiles) * (1 - overlap))
    y_win = int(img.shape[0]/float(n_tiles) * (1 - overlap))
    rmse_list = [0.]
    psnr_list = [0.]
    hist_list = [0.]

    for x0 in range(0, img.shape[1], x_win):
        x1 = x0 + x_win
        if x1 > img.shape[1] - 1:
            continue
        for y0 in range(0, img.shape[0], y_win):
            y1 = y0 + y_win
            if y1 > img.shape[0] - 1:
                continue

            a = img[y0:y1, x0:x1]

            for x00 in range(x0, img.shape[1], x_win):
                x11 = x00 + x_win
                if x11 > img.shape[1] - 1:
                    continue
                for y00 in range(y0, img.shape[0], y_win):
                    y11 = y00 + y_win
                    if y11 > img.shape[0] - 1:
                        continue

                    b = img[y00:y11, x00:x11]
                    out = compare(a, b, do_psnr, do_rmse, do_hist)
                    # TODO make output nicer to read
                    if do_rmse is True:
                        rmse_list.append(out[0])
                    if do_psnr is True:
                        psnr_list.append(out[-2])
                    if do_hist is True:
                        hist_list.append(out[-1])
    rmse_list.pop(0)
    psnr_list.pop(0)
    hist_list.pop(0)
    out = [0.]
    if do_rmse is True:
        r_list = np.array(rmse_list)
        mean_rmse = np.nanmean(r_list)
        var_rmse = np.nanvar(r_list)
        out.append(mean_rmse)
        out.append(var_rmse)
    if do_psnr is True:
        p_list = np.array(psnr_list)
        mean_psnr = np.nanmean(p_list)
        var_psnr = np.nanvar(p_list)
        out.append(mean_psnr)
        out.append(var_psnr)
    if do_hist is True:
        h_list = np.array(hist_list)
        mean_hist = np.nanmean(h_list)
        var_hist = np.nanvar(h_list)
        out.append(mean_hist)
        out.append(var_hist)
    out.pop(0)
    return out


@jit(nopython=True)
def otsu_heterogeneity(image):
    # numbafied directly (while having removed 'testing', (whoops) from skimage source code
    hist, bin_edges = np.histogram(image.ravel(), bins=256)
    hist = hist.astype(np.float32)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    binary = image > threshold

    het = binary.sum() / float(binary.shape[1] * binary.shape[0])
    return het


@jit(nopython=True)
def global_compare(img, do_mean, do_var, do_otsu):
    # global parameters
    out = [0.]
    if do_mean is True:
        m = np.nanmean(img)
        out.append(m)
    if do_var is True:
        v = np.nanvar(img)
        out.append(v)
    if do_otsu is True:
        o = otsu_heterogeneity(img)
        out.append(o)
    # TODO add particle passage (h, v)
    out.pop(0)
    return out


@jit(nopython=True)
def make_similarity_matrix(features):
    sim = np.zeros(shape=(features.shape[0], features.shape[0]))
    for i in range(0, features.shape[0]):
        for j in range(i, features.shape[0]):
            diff = np.sum(np.abs(features[i, :] - features[j, :]))/ float(features.shape[1])
            sim[i, j] = diff
            sim[j, i] = diff
        if i % 100 == 0:
            print("Generated", i, "of", features.shape[0], "similarity matrix entries")

    return sim


def cluster_results(similarity_matrix, n, save_to_folders=False, save_list=True):
    out_folder = "Clustered Images"

    model = MiniBatchKMeans(n_clusters=n, verbose=1).fit(similarity_matrix.values)
    labels = model.labels_
    out_df = pd.DataFrame(labels, index=similarity_matrix.index)
    out_df.rename(columns={0: "Cluster"}, inplace=True)
    if save_list is True:
        out_df.to_csv("cluster_labels.csv")

    if save_to_folders is True:
        if os.path.isdir(out_folder):
            shutil.rmtree(out_folder)
        os.mkdir(out_folder)

        for i in tqdm(range(n), desc="Moving Images to cluster folders"):
            os.mkdir("%s/%i" % (out_folder, i))

            sub = out_df[out_df["Cluster"] == i]
            names = sub.index.values
            for j in names:
                base = os.path.basename(j)
                shutil.copy(j, "%s/%i/%s" % (out_folder, i, base))
    return


def search_matches(similarity_matrix, query, results):
    try:
        out = np.argsort(similarity_matrix.loc[query].values)[1:results + 1]
    except (KeyError, TypeError) as e:
        raise UserWarning(
            "%s was not found in the index, make sure that the name you typed in is correct..." % query)

    print("The best %i matches are: " % results)
    for i in out:
        print(similarity_matrix.index.values[i])


class SimilarityGenerator:
    def __init__(self, img_list, params):
        pickle.dump([False], open("processing.pkl", "wb"))
        self.img_list = img_list
        if len(params) == 1 or img_list is None:
            # LOAD SIMILARITY
            similarity_matrix = pd.read_csv("similarity_matrix.csv")
            cluster_results(similarity_matrix, n=params[0])
        self.n_tiles = params[0]
        self.overlap = params[1]
        self.do_mean = params[2]
        self.do_var = params[3]
        self.do_otsu = params[4]
        self.do_psnr = params[5]
        self.do_rmse = params[6]
        self.do_hist = params[7]
        self.do_multi = params[8]

        self.features_dict = self.extract_features()
        self.features_matrix = self.convert_dict_to_matrix()
        self.features_matrix = self.scale_columns()
        similarity_matrix = make_similarity_matrix(self.features_matrix.values)
        pd.DataFrame(similarity_matrix, index=self.features_matrix.index).to_csv("similarity_matrix.csv")

        pickle.dump([True], open("processing.pkl", "wb"))
    @staticmethod
    def load_img_as_array(img):
        img = Image.open(img, "r")
        img = ImageOps.grayscale(img)  # makes it greyscale
        img = np.asarray(img.getdata(), dtype=np.float64).reshape((img.size[1], img.size[0]))
        return img

    def extract_loop(self, chunk, process, features_dict):
        for i in tqdm(chunk, desc="Extracting Features, process %i" % process):
            img = self.load_img_as_array(i)
            out = global_compare(img, self.do_mean, self.do_var, self.do_otsu)
            out += compare_img_tiles(img, self.n_tiles, self.overlap, self.do_psnr, self.do_rmse, self.do_hist)

            features_dict[i] = out

        return features_dict

    def extract_features(self):

        if self.do_multi <= 1:
            features_dict = dict()
            features_dict = self.extract_loop(self.img_list, 1, features_dict)

        else:
            if self.do_multi == -1:
                self.do_multi = os.cpu_count()
            if self.do_multi > os.cpu_count():
                self.do_multi = os.cpu_count()
            manager = multiprocessing.Manager()

            return_dict = manager.dict()
            sub_lists = np.array_split(self.img_list, self.do_multi)
            processes = list()
            for i, chunk in enumerate(sub_lists):
                p = multiprocessing.Process(target=self.extract_loop, args=(chunk, i, return_dict))
                processes.append(p)

            for p in processes:
                p.start()
            for p in processes:
                p.join()

            features_dict = dict()
            for r in return_dict.keys():
                features_dict[r] = return_dict[r]
        return features_dict

    def convert_dict_to_matrix(self):
        df = pd.DataFrame.from_dict(self.features_dict, orient="index")
        return df

    def scale_columns(self):
        df = self.features_matrix
        for col in df.columns:
            vals = df[col]
            vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
            df[col] = vals
        return df

