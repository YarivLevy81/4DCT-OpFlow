"""
Converts a folder of dcm images consisting a series of 3d images to numpy arrays



.. usage:

   change the variables dir folder path, first img num, last an patient num

.. note:

   Uses numpy, pydicom and matplotlib.

"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def images_to_3dnump(dir_name, img_num, plot=False):
    # load the DICOM files
    files = []
    dir_name_for = dir_name + "/IM-00{:02}*.dcm".format(img_num)
    print("glob: {}".format(dir_name_for))
    for fname in glob.glob(dir_name_for, recursive=False):
        #       print("loading: {}".format(fname))
        files.append(pydicom.dcmread(fname))

    print("file count: {}".format(len(files)))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda sl: sl.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]
    voxel_size = np.ndarray(shape=3, buffer=np.array([ps[0], ps[1], ss]), dtype=float)

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    # plot 3 orthogonal slices
    if plot:
        a1 = plt.subplot(2, 2, 1)
        plt.imshow(img3d[:, :, img_shape[2] // 2])
        a1.set_aspect(ax_aspect)

        a2 = plt.subplot(2, 2, 2)
        plt.imshow(img3d[:, img_shape[1] // 2, :])
        a2.set_aspect(sag_aspect)

        a3 = plt.subplot(2, 2, 3)
        plt.imshow(img3d[img_shape[0] // 2, :, :].T)
        a3.set_aspect(cor_aspect)
        plt.title(img_num)
        plt.show()

    return img3d, voxel_size


def nd_list_to_4d(nd_arrays) -> np.ndarray:
    mat_shape = list(nd_arrays[0].shape)
    mat_shape.insert(0, len(nd_arrays))
    mat4d = np.zeros(mat_shape)
    for i, img3d in enumerate(nd_arrays):
        mat4d[i, :, :, :] = img3d
    return mat4d


def convert_matrix_folder_to_4dnpz_file(
        dir_path, trgt_path, pat_num, first_img, end_img, onefile=True):
    nd_arrs = []
    for img_num in range(first_img, end_img + 1):
        img3d, voxel_dim = images_to_3dnump(dir_path, img_num)
        nd_arrs.append(img3d)
    mat_4d = nd_list_to_4d(nd_arrs)
    if onefile:
        name = f'mat_{pat_num}_{mat_4d.shape}.npz'
        with open(trgt_path + name, 'wb') as out:
            np.savez_compressed(out, data=mat_4d, vox_dim=voxel_dim)

    else:
        for i in range(len(nd_arrs)):
            with open(trgt_path + f"{i + 1}.npy", 'wb') as out:
                np.save(out, nd_arrs[i])


def npz_to_ndarray_and_vox_dim(filename) -> np.ndarray:
    with np.load(filename) as npzfile:
        mat3d = npzfile['data']
        vox_dim = npzfile['vox_dim']
        return mat3d, vox_dim


def validation_to_npz(target_dir_name, filename, im1, im1_vox, im2, im2_vox, flow):
    with open(target_dir_name + f'/mat_{filename}_valid {im1.shape}.npz', 'wb') as outfile:
        np.savez_compressed(outfile, img1_data=im1, img1_vox_dim=im1_vox, img2_data=im2, img2_vox_dim=im2_vox,
                            flow=flow)


def npz_valid_to_ndarrays_flow_vox(filename):
    with np.load(filename) as npzfile:
        img1_data = npzfile['img1_data']
        img1_vox_dim = npzfile['img1_vox_dim']
        img2_data = npzfile['img2_data']
        img2_vox_dim = npzfile['img2_vox_dim']
        flow12 = npzfile['flow']
        return (img1_data, img1_vox_dim), (img2_data, img2_vox_dim), flow12


def mat4d_to_mat3d_fold(mat4d, vox_dim, parent_dir_name, trgt_dir_name):
    directory = parent_dir_name + '/' + trgt_dir_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(mat4d.shape[0]):
        mat3d = mat4d[i, :, :, :]
        with open(directory + f'/mat_{i}_{mat3d.shape}.npz', 'wb') as outfile:
            np.savez_compressed(outfile, data=mat3d, vox_dim=vox_dim)


def mat4dfold_to_mat3dfolds(dir_name):
    files = []
    for fname in glob.glob(dir_name + "*.npz", recursive=False):
        # print("loading: {}".format(fname))
        files.append(fname)
    for file in files[47:]:
        name = file.split("_")
        fname = name[2] + '_'
        if len(name) != 4:
            for i in range(3, len(name) - 1):
                fname += (name[i] + '_')
        print(fname)
        mat4d, vox_dim = npz_to_ndarray_and_vox_dim(file)
        print("mat loaded")
        mat4d_to_mat3d_fold(mat4d, vox_dim, dir_name, fname[:-1])


dir_folder_path = "/mnt/storage/datasets/4DCT/041516 New Cases/4/Anonymized - 4719590/Ctacor/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 10/"
target_path = "/mnt/storage/datasets/4DCT/041516 New Cases/Nd_arrays/"
patient_num = "4"
first_img_number = 9
last_img_number = 28
mats = []
# mat4dfold_to_mat3dfolds(target_path)
# mats.append(("21","/",7,26))
# mats.append(("22","/",7,29))
# mats.append(("24","/",11,29))
# mats.append(("26","/mnt/storage/datasets/4DCT/041516 New Cases/26/Anonymized - 3080856/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% Matrix 256 - 8/",7,8))

# mats.append(("40_2","/mnt/storage/datasets/4DCT/041516 New Cases/40.0/Anonymized - 4388157/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 10/",1,20))


# mats.append(("1","/",11,29))


for mat in mats:
    convert_matrix_folder_to_4dnpz_file(mat[1], target_path, mat[0], mat[2], mat[3])

# convert_matrix_folder_to_4dnpz_file(dir_folder_path,target_path, patient_num, first_img_number, last_img_number)


""" 
mats.append(("1","/mnt/storage/datasets/4DCT/041516 New Cases/1/Anonymized - 3084943/Ctacoc/CorVein_Bi 1.5 B25f MPR 5-95% Matrix 256 - 14/",11,29))
mats.append(("2","/mnt/storage/datasets/4DCT/041516 New Cases/2/Anonymized - 3863450/Ctacoc/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 9/",8,27))
mats.append(("3","/mnt/storage/datasets/4DCT/041516 New Cases/3/Anonymized - 1221351/Ctacoc-3D/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 10/",9,28))
mats.append(("3_mono","/mnt/storage/datasets/4DCT/041516 New Cases/3/Anonymized - 1221351/Ctacoc-3D/CorVein_Mono 1.5 B25f MPR 0-95% Matrix 256 - 11/",29,48))
mats.append(("4","/mnt/storage/datasets/4DCT/041516 New Cases/4/Anonymized - 4719590/Ctacor/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 10/",9,28))
mats.append(("4_mono","/mnt/storage/datasets/4DCT/041516 New Cases/4/Anonymized - 4719590/Ctacor/CorVein_Mono 1.5 B25f MPR 0-95% Matrix 256 - 11/",29,48))
mats.append(("5","/mnt/storage/datasets/4DCT/041516 New Cases/5/Anonymized - 3943804/Ctacor/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 10/",9,28))
mats.append(("5_mono","/mnt/storage/datasets/4DCT/041516 New Cases/5/Anonymized - 3943804/Ctacor/CorVein_Mono 1.5 B25f MPR 0-95% Matrix 256 - 11/",29,48))
mats.append(("6","/mnt/storage/datasets/4DCT/041516 New Cases/6/Anonymized - 4703259/Ctacor/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 8/",7,26))
mats.append(("6_mono","/mnt/storage/datasets/4DCT/041516 New Cases/6/Anonymized - 4703259/Ctacor/CorVein_Mono 1.5 B25f MPR 0-95% Matrix 256 - 9/",27,46))
mats.append(("7","/mnt/storage/datasets/4DCT/041516 New Cases/7/Anonymized - 4681521/Ctacor/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 4/",3,22))
mats.append(("7_mono","/mnt/storage/datasets/4DCT/041516 New Cases/7/Anonymized - 4681521/Ctacor/CorVein_Mono 1.5 B25f MPR 0-95% Matrix 256 - 9/",25,44))
mats.append(("8","/mnt/storage/datasets/4DCT/041516 New Cases/8/Anonymized - 3645265/Ctacor/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 4/",3,22))
mats.append(("8_mono","/mnt/storage/datasets/4DCT/041516 New Cases/8/Anonymized - 3645265/Ctacor/CorVein_Mono 1.5 B25f MPR 0-95% Matrix 256 - 14/",29,48))
mats.append(("9","/mnt/storage/datasets/4DCT/041516 New Cases/9/Anonymized - 4887071/Ctacor/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 9/",7,26))
mats.append(("9_2","/mnt/storage/datasets/4DCT/041516 New Cases/9/Anonymized - 4887071/Ctacor/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 10/",27,46))
mats.append(("10","/mnt/storage/datasets/4DCT/041516 New Cases/10/Anonymized - 3721180/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 10/",9,28))
mats.append(("11","/mnt/storage/datasets/4DCT/041516 New Cases/11/Anonymized - 3851090/Ctacor/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 8/",7,26))
mats.append(("12","/mnt/storage/datasets/4DCT/041516 New Cases/12/Anonymized - 1124707/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 9/",8,27))
mats.append(("12_mono","/mnt/storage/datasets/4DCT/041516 New Cases/12/Anonymized - 1124707/Ctacoc/DS_CorCTAMono 1.5 B25f 0-95% Matrix 256 - 10/",28,47))
mats.append(("13","/mnt/storage/datasets/4DCT/041516 New Cases/13/Anonymized - 4903935/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 8/",7,26))
mats.append(("14","/mnt/storage/datasets/4DCT/041516 New Cases/14/Anonymized - 4071156/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 9/",8,27))
mats.append(("15","/mnt/storage/datasets/4DCT/041516 New Cases/15/Anonymized - 3390144/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 9/",7,26))
mats.append(("16","/mnt/storage/datasets/4DCT/041516 New Cases/16/Anonymized - 1910812/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 8/",7,26))
mats.append(("17","/mnt/storage/datasets/4DCT/041516 New Cases/17/Anonymized - 3336452/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 10/",9,28))
mats.append(("18","/mnt/storage/datasets/4DCT/041516 New Cases/18/Anonymized - 1946854/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 8/",7,26))
mats.append(("19","/mnt/storage/datasets/4DCT/041516 New Cases/19/Anonymized - 1834896/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 8/",7,26))
mats.append(("20","/mnt/storage/datasets/4DCT/041516 New Cases/20/Anonymized - 859733/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 12/",9,28))
mats.append(("23","/mnt/storage/datasets/4DCT/041516 New Cases/23/Anonymized - 2090493/Ctacoc/DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 8/",7,26))
mats.append(("27","/mnt/storage/datasets/4DCT/041516 New Cases/27/Anonymized - 3689411/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("28","/mnt/storage/datasets/4DCT/041516 New Cases/28/Anonymized - 2889363/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("29","/mnt/storage/datasets/4DCT/041516 New Cases/29/Anonymized - 2247170/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("30","/mnt/storage/datasets/4DCT/041516 New Cases/30/Anonymized - 4435086/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("25","/mnt/storage/datasets/4DCT/041516 New Cases/25/Anonymized - 1347256/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% Matrix 256 - 11/",9,20))
mats.append(("25_2","/mnt/storage/datasets/4DCT/041516 New Cases/25/Anonymized - 1347256/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% Matrix 256 - 11/",22,28))
mats.append(("31","/mnt/storage/datasets/4DCT/041516 New Cases/31/Anonymized - 1670653/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("32","/mnt/storage/datasets/4DCT/041516 New Cases/32/Anonymized - 751102/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("33","/mnt/storage/datasets/4DCT/041516 New Cases/33/Anonymized - 3953561/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("34","/mnt/storage/datasets/4DCT/041516 New Cases/34/Anonymized - 3962462/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("35","/mnt/storage/datasets/4DCT/041516 New Cases/35/Anonymized - 2772063/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 11/",9,28))
mats.append(("36","/mnt/storage/datasets/4DCT/041516 New Cases/36/Anonymized - 3932529/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 6/",5,24))
mats.append(("37","/mnt/storage/datasets/4DCT/041516 New Cases/37/Anonymized - 3587363/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 7/",6,25))
mats.append(("38","/mnt/storage/datasets/4DCT/041516 New Cases/38/Anonymized - 2117853/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 9/",8,27))
mats.append(("39","/mnt/storage/datasets/4DCT/041516 New Cases/39/Anonymized - 5102426/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("40_non_mat","/mnt/storage/datasets/4DCT/041516 New Cases/40/Anonymized - 4388157/Ctacoc/DS_CorCTABi 0.75 B26f 0-95% NONMATRIX - 12/",77,88))
mats.append(("40","/mnt/storage/datasets/4DCT/041516 New Cases/40/Anonymized - 4388157/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 10/",17,56))
mats.append(("40_mono","/mnt/storage/datasets/4DCT/041516 New Cases/40/Anonymized - 4388157/Ctacoc/DS_CorCTAMono 1.5 B26f 0-95% - 11/",57,76))
mats.append(("40_2_non_mat","/mnt/storage/datasets/4DCT/041516 New Cases/40.0/Anonymized - 4388157/Ctacoc/DS_CorCTABi 0.75 B26f 0-95% NONMATRIX - 12/",41,60))
mats.append(("40_2_mono","/mnt/storage/datasets/4DCT/041516 New Cases/40.0/Anonymized - 4388157/Ctacoc/DS_CorCTAMono 1.5 B26f 0-95% - 11/",21,40))
mats.append(("41","/mnt/storage/datasets/4DCT/041516 New Cases/41/Anonymized - 2339528/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("42","/mnt/storage/datasets/4DCT/041516 New Cases/42/Anonymized - 5171704/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 11/",10,29))
mats.append(("42_2","/mnt/storage/datasets/4DCT/041516 New Cases/42.0/Anonymized - 5171704/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 11/",6,25))
mats.append(("43","/mnt/storage/datasets/4DCT/041516 New Cases/43/Anonymized - 3184349/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("43_2","/mnt/storage/datasets/4DCT/041516 New Cases/43.0/Anonymized - 3184349/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",3,22))
mats.append(("44","/mnt/storage/datasets/4DCT/041516 New Cases/44/Anonymized - 4669044/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("45","/mnt/storage/datasets/4DCT/041516 New Cases/45/Anonymized - 4400607/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 8/",7,26))
mats.append(("46","/mnt/storage/datasets/4DCT/041516 New Cases/46/Anonymized - 948015/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 10/",9,28))
mats.append(("47","/mnt/storage/datasets/4DCT/041516 New Cases/47/Anonymized - 2395933/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 9/",8,27))
mats.append(("48_non_mat","/mnt/storage/datasets/4DCT/041516 New Cases/48/Anonymized - 1872695/Ctacoc/DS_CorCTABi 0.75 B26f 0-95% NONMATRIX - 12/",49,68))
mats.append(("48","/mnt/storage/datasets/4DCT/041516 New Cases/48/Anonymized - 1872695/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 10/",9,28))
mats.append(("48_mono","/mnt/storage/datasets/4DCT/041516 New Cases/48/Anonymized - 1872695/Ctacoc/DS_CorCTAMono 1.5 B26f 0-95% - 11/",29,48))
mats.append(("49","/mnt/storage/datasets/4DCT/041516 New Cases/49/Anonymized - 2485709/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 10/",9,28))
mats.append(("50","/mnt/storage/datasets/4DCT/041516 New Cases/50/Anonymized - 4563861/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 12/",10,28))
mats.append(("51","/mnt/storage/datasets/4DCT/041516 New Cases/51/Anonymized - 2104942/Ctacoc/DS_CorCTABi 1.5 B26f 0-95% - 11/",10,29))
mats.append(("52","/mnt/storage/datasets/4DCT/041516 New Cases/52/Anonymized - 5161372/Ctacoc/CINE 0 - 95 % - 10/",9,28))
mats.append(("53","/mnt/storage/datasets/4DCT/041516 New Cases/53/Anonymized - 913279/Ctacoc/CINE 0 - 95 % BISEGMENT - 7/",6,25))
mats.append(("54","/mnt/storage/datasets/4DCT/041516 New Cases/54/Anonymized - 3598522/Ctacoc/CINE 0 - 95 % BISEGMENT - 11/",10,29))
mats.append(("55","/mnt/storage/datasets/4DCT/041516 New Cases/55/Anonymized - 3758993/Ctacoc-3D/CINE 0 - 95 % BISEGMENT - 10/",9,28))
mats.append(("56_2","/mnt/storage/datasets/4DCT/041516 New Cases/56/Anonymized - 5063582/Ctacom-3D/CINE 200-700ms - 8/",4,14))
mats.append(("56","/mnt/storage/datasets/4DCT/041516 New Cases/56/Anonymized - 5063582/Ctacom-3D/CINE 180-480 ms - 9/",15,29))
 """
