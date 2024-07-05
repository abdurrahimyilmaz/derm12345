#%% -*- coding: utf-8 -*-
# For original dataset

import numpy as np
from tqdm import tqdm
import os
import cv2
from skimage.transform import resize

#%%
os.getcwd()
Dir = "YOUR PATH" # your path

os.chdir(Dir)
path = os.getcwd()
os.listdir(Dir) # see melanocytic and nonmelanocytic folders

#%%
# move between folders

not_subfolders = ["melanocytic", "indeterminate", "nonmelanocytic", "benign", "malignant", "banal", "dysplastic", "recurrent", "compound", "dermal", "junctional", "lentigo", "melanoma", "keratinocytic", "skin_appendages", "vascular"]
counter = 0

paths = []

for root, dirs, files in os.walk(path):
    for dir in dirs:
        if ("melanocytic" in root or "nonmelanocytic" in root) and any(i == dir for i in not_subfolders) == False:
            paths.append(os.path.join(root, dir))

            #fullpaths.append(root + "/" + file)
            #filepaths.append(file)

print("Number of folders: ", len(paths)) # must be 38

#%%
image_width, image_height, image_depth = 299, 299, 3

def get_data(paths):
    X = []
    y = []
    counter = 0
    for nextDir in paths:
        if os.path.basename(os.path.normpath(nextDir)) ==   "acral_compound_banal":
            label = 0 # dont change start from 0 always for straightforward ai training
        elif os.path.basename(os.path.normpath(nextDir)) == "compound_banal":
            label = 1 
        elif os.path.basename(os.path.normpath(nextDir)) == "congenital_compound_banal":
            label = 2
        elif os.path.basename(os.path.normpath(nextDir)) == "miescher_compound_banal":
            label = 3
        elif os.path.basename(os.path.normpath(nextDir)) == "blue_dermal_banal":
            label = 4
        elif os.path.basename(os.path.normpath(nextDir)) == "dermal_banal":
            label = 5
        elif os.path.basename(os.path.normpath(nextDir)) == "acral_junctional_banal":
            label = 6
        elif os.path.basename(os.path.normpath(nextDir)) == "congenital_junctional_banal":
            label = 7
        elif os.path.basename(os.path.normpath(nextDir)) == "junctional_banal":
            label = 8
        elif os.path.basename(os.path.normpath(nextDir)) == "acral_compound_dysplastic":
            label = 9
        elif os.path.basename(os.path.normpath(nextDir)) == "compound_dysplastic":
            label = 10
        elif os.path.basename(os.path.normpath(nextDir)) == "congenital_compound_dysplastic":
            label = 11
        elif os.path.basename(os.path.normpath(nextDir)) == "acral_junctional_dysplastic":
            label = 12
        elif os.path.basename(os.path.normpath(nextDir)) == "junctional_dysplastic":
            label = 13
        elif os.path.basename(os.path.normpath(nextDir)) == "spitz_reed_junctional_dysplastic":
            label = 14
        elif os.path.basename(os.path.normpath(nextDir)) == "recurrent_dysplastic":
            label = 15
        elif os.path.basename(os.path.normpath(nextDir)) == "ink_spot_lentigo":
            label = 16
        elif os.path.basename(os.path.normpath(nextDir)) == "lentigo_simplex":
            label = 17
        elif os.path.basename(os.path.normpath(nextDir)) == "solar_lentigo":
            label = 18
        elif os.path.basename(os.path.normpath(nextDir)) == "acral_nodular":
            label = 19
        elif os.path.basename(os.path.normpath(nextDir)) == "acral_lentiginious":
            label = 20
        elif os.path.basename(os.path.normpath(nextDir)) == "lentigo_maligna":
            label = 21
        elif os.path.basename(os.path.normpath(nextDir)) == "lentigo_maligna_melanoma":
            label = 22
        elif os.path.basename(os.path.normpath(nextDir)) == "malignant_melanoma":
            label = 23
        elif os.path.basename(os.path.normpath(nextDir)) == "lichenoid_keratosis":
            label = 24
        elif os.path.basename(os.path.normpath(nextDir)) == "seborrheic_keratisos":
            label = 25
        elif os.path.basename(os.path.normpath(nextDir)) == "dermatofibroma":
            label = 26
        elif os.path.basename(os.path.normpath(nextDir)) == "hemangioma":
            label = 27
        elif os.path.basename(os.path.normpath(nextDir)) == "lymphangioma":
            label = 28
        elif os.path.basename(os.path.normpath(nextDir)) == "pyogenic_granuloma":
            label = 29
        elif os.path.basename(os.path.normpath(nextDir)) == "actinic_keratosis":
            label = 30
        elif os.path.basename(os.path.normpath(nextDir)) == "basal_cell_carcinoma":
            label = 31
        elif os.path.basename(os.path.normpath(nextDir)) == "bowen_disease":
            label = 32
        elif os.path.basename(os.path.normpath(nextDir)) == "cutaneous_horn":
            label = 33
        elif os.path.basename(os.path.normpath(nextDir)) == "mammary_paget_disease":
            label = 34
        elif os.path.basename(os.path.normpath(nextDir)) == "squamous_cell_carcinoma":
            label = 35
        elif os.path.basename(os.path.normpath(nextDir)) == "dermatofibrosarcoma_protuberans":
            label = 36
        elif os.path.basename(os.path.normpath(nextDir)) == "kaposi_sarcoma":
            label = 37
            
        # this label part can be extended for your use case and folder structure         

        for file in tqdm(os.listdir(nextDir)):
            img = cv2.imread(nextDir + '/' + file)
            if img is None:
                print("Error: ", nextDir + '/' + file) # for debugging to see which files are not read
            elif img is not None:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = resize(img, (image_width, image_height, image_depth)) #Change X, Y, Z parameters according to your deep learning model input.
                #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                img = np.asarray(img)
                X.append(img)
                y.append([label, file])
                counter += 1

    X = np.asarray(X)
    y = np.asarray(y)
    print("Number of images: ", counter) # must be 12345
    return X, y
#%%
X, y = get_data(paths)
print("Your np arrays has been successfully generated.")
#%%
os.chdir("YOUR PATH") # your path
file_name = "derm12345"
#np.save(f"{file_name}_X.npy", X)
np.save(f"{file_name}_y.npy", y)
print("Your np arrays has been successfully saved.")

# %%
from numpy import load
from tqdm import tqdm

path = "YOUR PATH"
x_data = load(os.path.join(path, "derm12345_X.npy"))
y_data = load(os.path.join(path, "derm12345_y.npy"))

#%%
# plot first few images
import matplotlib.pyplot as plt
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_data[i])
plt.show()

del x_data

# %%
# create pandas dataframe and merge with metadata
import pandas as pd
import numpy as np

#read csv file
metadata = pd.read_csv("YOUR FILE PATH")
#print(metadata.head())

dummy_column = np.ones((y_data.shape[0], 14))
y_data = np.c_[y_data, dummy_column]

# %%

for i in tqdm(range(y_data.shape[0])):
    filtered_metadata = metadata[metadata["image_id"].str.contains(y_data[i][1][:-4])] # search in metadata without .jpg
    y_data[i][2:] = filtered_metadata.values
    
# %%

y_data = pd.DataFrame(y_data, columns=["raw_label", "file_name", "image_id", "patient_id", "location", "diagnosis_confirm_type", "lesion_count_for_patient", "license", "width", "height", "image_type", "super_class", "malignancy", "main_class_1", "main_class_2", "sub_class"])
# %%

y_data.to_csv("YOUR CSV PATH", index=False)
# %%
