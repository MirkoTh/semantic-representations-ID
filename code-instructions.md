# Individual Differences in Object Representations

**Instructions for how to run the code, reproduce the results, and run Study 2.**

Please first read the following points carefully before using the code:

- Running the model may take a lot of time. We therefore also provide the resulting files from the modeling scripts, so you can analyze the results without actually running the model.
- In the following instructions and explanations, the root folder of the project is represented by "~/"
- If you want to run Study 2 in jatos (without running the models and analyzing them), we provide the full code repo under ~/study-only/



Modeling and analysis code is provided in R and python; the experiment is programmed with jsPsych and custom java script, html, and css code. Before starting, make sure to carry out the following steps:

1. a local copy of jatos (www.jatos.org). This will allow you to run the experiment locally.
2. the rutils R package from the main author (available under github.com/MirkoTh/rutils). This will allow you to run all the R scripts.
3. download the files from the source studies relevant for the current project.
    - From https://osf.io/z2784/files/osfstorage, download the following files: labels.mat, words.mat, unique_id.txt.
    - From https://osf.io/f5rn6/overview download the file called triplets_large_final_correctednc_correctedorder.csv.

    --> Place all four downloaded files into ~/data/

    - From https://osf.io/jum2f/overview download the images from the source study as a zip file. Place the unzipped file called "images" into the ~/data/ folder.


## Study 1
Run the script ~/R/things-triplets.R to create the necessary data files for running the model. Note that this script also writes a file containing the "diagnostic triplets", i.e. those, which have been observed multiple times. This is going to be important for creating the fixed triplet set used in Study 2.

After that, you are ready to run the pytorch models, i.e. the weighted-embedding models.

### Hyperparameter search on lambda
Run the python file ~/initialize-model-highdim.py
If you want to obtain the results faster, consider splitting the models, e.g., by just running one lambda value on one computer, and the other lambda values on different computers.

### Larger individual-differences effects over dimensionality
Run the python file ~/initialize-model-improvement-dimensionality.py

### Split-half reliabilities
Run ~/initialize-model-splithalf-reliability-cc.py and ~/initialize-model-splithalf-reliability.py

After you have run the models, you can analyze the results with a set of jupyter notebooks, which are listed in the following.

### Analyze the hyperparameter search
~/analyze-highdim-model.ipynb

### Analyze how dimensionality affects the improvement due to the individual-differences components
~/dimensionality-accuracy-improvement.ipynb

### Calculate the split-half reliabilities
~/split-half-reliability.ipynb


## Study 2
First, we create the triplet set used in Study 2. Then, we run the study using jatos.

We saved the IDs of the used triplets in the file ~/data/triplets-delta_USED_STUDY2.csv. If you want to replicate our results using the same 440 triplets we used, then rename this triplet file to ~/data/triplets-delta.csv and skip the step "### Model deltas" below. Note, however, that unfortunately this triplet set is not re-created to 100% when running the following code below as the triplet set had been saved earlier in the project. The reasons can be the use of different seed values for running the model, modifications in the model architecture (e.g., with/without by-participant decision scaling factors, the used dimensionality), etc..

When using the model reported in the manuscript with freely varying dimensional weights (i.e., not modeled as random effects), the overlap with the original triplet set, however, is substantial. For the model-based part of the triplet set (i.e., half of the used 440 triplets), the overlap is 166 out of 220 triplets, 171 out of 220 triplets, and 163 out of 220 triplets for dimensionalities 25, 30, and 35, respectively.


### Model deltas
Run the jupyter notebook ~/analyze-model-deltas.ipynb to calculate the differences in prediction accuracy between the average representations and the idiosyncratically weighted representations. This notebook saves the file ~/data/triplets-delta.csv.

### Create triplet Set
To create the fixed set of 440 triplets, run the jupyter notebook create-triplet-set-ipynb. This notebook loads the file ~/data/triplets-delta.csv, and creates the triplet set consisting of 440 triplets (220 model-based and 220 random) and saves the necessary files for running the study in jatos. As indicated above, if you run from scratch, the resulting set will deviate from the set used in Study 2.


### Run the study
We provide a separate folder with all the code and files necessary to run the study in jatos. For details on jatos, visit www.jatos.org. In case you want to re-create the study from scratch using the results from the analyses, you would have to move all the relevant files from the source folder ~/ to the respective jatos folder (study_assets_root) on your computer.

### Load the data
Run the following two R scripts sequentially: ~/exclusion-criteria.R, ~/concatenate-ooo-old-new.R. These scripts filter the data according to the exclusion criteria, and concatenate our new results with the results from the source study. Note that we do not provide raw data files from prolific, but only data files with hashed prolific ids.

### Analyze dimensionality 35
Run the jupyter notebook ~/analyze-combined-data-finaldym.ipynb

### Analyze all 12 dimensionalities
Run the jupyter notebook ~/analyze-combined-data-alldims.ipynb

### Predict dimensional weightings using self-report text responses about work history and general interests
Run the jupyter notebook ~/predict-dims-by-interests.ipynb

## Plot all the figures of the MS

After running all the models, analyses, etc. from above, the relevant data sets have been saved. Now, you can plot all the results from the MS running the following two scripts:
1. R script: ~/R/plot-figures-ms.R
2. jupyter notebook: ~/plot-figures-ms.ipynb

Note that the overview figures of the study and the first result figure have been manually put together from the resulting figures.
