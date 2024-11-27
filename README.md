DERM12345: A Large, Multisource Dermatoscopic Skin Lesion Dataset with 38 Subclasses<br>
DERM12345 dataset contains 12345 dermatoscopic images with 5 super classes, 15 main classes, and 38 subclasses.

This repository is to share useful sources to process DERM12345 dataset for data analysis and training AI.

Last updated: 27/11/2024<br>

Environment:<br>
python 3.11.5<br>
jupyterlab 3.6.3<br>
beautifulsoup4 4.12.2<br>
numpy 1.24.3<br>
pandas 2.0.3<br>
imagesize 1.4.1<br>

Contains two main folders:<br>
1-data: This folder is to share Python scripts for development of data arrays for AI training. <br>
2-ai_training: This folder is to train baseline deep learning models. <br>

Taxonomy Tree<br>
-Melanocytic Benign<br>
a---Banal Compound<br>
1-----Acral (acb)<br>
2-----Congenital (ccb)<br>
3-----Miescher (mcb)<br>
4-----Banal Compound Nevus (cb)<br>
b---Banal Dermal<br>
1-----Blue (bdb)<br>
2-----Banal Dermal Nevus (db)<br>
c---Banal Junctional<br>
1-----Acral (ajb)<br>
2-----Congenital (cjb)<br>
3-----Banal Junctional Nevus (jb)<br>
d---Dysplastic Compound<br>
1-----Acral (acd)<br>
2-----Congenital (ccd)<br>
3-----Dysplastic Compound Nevus (cd)<br>
e---Dysplastic Junctional<br>
1-----Acral (ajd)<br>
2-----Spitz/Reed (srjd)<br>
3-----Dysplastic Junctional Nevus (jd)<br>
f---Dysplastic Recurrent<br>
1-----Dysplastic Recurrent Nevus (rd)<br>
g---Lentigo<br>
1-----Ink Spot Lentigo (isl)<br>
2-----Lentigo Simplex (ls)<br>
3-----Solar Lentigo (sl)<br>
<br>
-Melanocytic Malignant<br>
a---Melanoma<br>
1-----Acral Nodular (anm)<br>
2-----Acral Lentiginious (alm)<br>
3-----Lentigo Maligna (lm)<br>
4-----Lentigo Maligna Melanoma (lmm)<br>
5-----Melanoma (mel)<br>
<br>
-Nonmelanocytic Benign<br>
a---Keratinocytic<br>
1-----Seborrheic Keratosis (sk)<br>
2-----Lichenoid Keratosis (lk)<br>
b---Fibro-histiocytic<br>
1-----Dermatofibroma (df)<br>
c---Vascular<br>
1-----Angiokeratoma (angk)
2-----Hemangioma (ha)
3-----Lymphangioma (la)
4-----Pyogenic Granuloma (pg)
5-----Spider Angioma (sa)
<br>
-Nonmelanocytic Indeterminate<br>
a---Keratinocytic<br>
1-----Actinic Keratosis (ak)<br>
<br>
-Nonmelanocytic Malignant<br>
a---Keratinocytic<br>
1-----Basal Cell Carcinoma (bcc)<br>
2-----Bowen's Disease (bd)<br>
3-----Cutaneous Horn (ch)<br>
4-----Mammary Paget Disease (mpd)<br>
5-----Squamous Cell Carcinoma (scc)<br>
b---Fibro-histiocytic<br>
1-----Dermatofibrosarcoma Protuberans (dfsp)<br>
c---Vascular<br>
1-----Kaposi Sarcoma (ks)<br>
