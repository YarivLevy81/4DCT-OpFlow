"""
==========================================================
Load CT slices and plot axial, sagittal and coronal images
==========================================================

This example illustrates loading multiple files, sorting them by slice
location, building a 3D image and reslicing it in different planes.

.. usage:

   reslice.py <glob>
   where <glob> refers to a set of DICOM image files.

   Example: python reslice.py "*.dcm". The quotes are needed to protect
   the glob from your system and leave it for the script.

.. note:

   Uses numpy and matplotlib.

   Tested using series 2 from here
   http://www.pcir.org/researchers/54879843_20060101.html
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import glob


def images_to_3dnump(dir_name, img_num, plot=False):
    # load the DICOM files
    files = []
    dir_name_for = dir_name + "\\IM-00{:02}*.dcm".format(img_num)
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

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    return img3d
    # save to npy file

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


def convert_matrix_folder_to_3dnpy_files(
        dir_path, pat_num, first_img, end_img, onefile=True):
    nd_arrs = []
    for img_num in range(first_img, end_img + 1):
        nd_arrs.append(images_to_3dnump(dir_path, img_num))
    if onefile:
        with open(dir_path + f'mat_{pat_num}_{len(nd_arrs)}.npz', 'w') as out:
            np.savez_compressed(out, nd_arrs)

    else:
        for i in range(len(nd_arrs)):
            with open(dir_path + f"{i + 1}.npy", 'w') as out:
                np.save(out, nd_arrs[i])


dir_folder_path = "/mnt/storage/datasets/4DCT/041516 New Cases"
dir_folder_path += "/2/Anonymized - 3863450"
dir_folder_path += "/Ctacoc/CorVein_Bi 1.5 B25f MPR 0-95% Matrix 256 - 9/"
patient_num = "2"
first_img_number = 8
last_img_number = 27

convert_matrix_folder_to_3dnpy_files(
    dir_folder_path, patient_num, first_img_number, last_img_number
)
