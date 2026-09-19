# Session 20: Machine Learning Applications

Runnable versions of the four code demos in the Session 20 deck, plus the sources for the take-home assignments.

## Scripts

| Script | Deck file | Data | Result to reproduce |
|---|---|---|---|
| `demo_housing_regression.py` | `ml_course_demo_regression_housing.tex` | California Housing (downloads once) | Linear and Ridge RMSE 0.7456, R2 0.5758; Lasso(0.1) RMSE 0.8244, R2 0.4814 |
| `demo_svm_digits.py` | `ml_course_demo_svm_digits.tex` | Digits (built in) | accuracy 0.9806, 7 errors out of 360 |
| `demo_customer_segments.py` | `ml_course_demo_clustering_customers.tex` | synthetic, seeded | inertia 720.0, 216.9, 87.7, 76.5, 65.2, 54.8, 48.2, 41.4, 37.3 |
| `assign_pca_digits.py` | `ml_course_assign_pca_digits.tex` | Digits (built in) | ARI 0.326, 0.483, 0.476 for 2, 10, 30 components |

Run any of them with `conda activate mlcoep && python <script>`. Figures are shown on screen, or saved as PNG files next to
the script on a machine without a display (the PNG files are git-ignored).

`fetch_california_housing` needs internet on the first run only; scikit-learn then keeps a copy in `~/scikit_learn_data`.

`assign_pca_digits.py` is an assignment: it runs Parts A to C as the slides show them and adds a starter for the remaining
tasks (variance explained, scree plot, cluster centers). Students should write their own answers to the questions first.

## Take-home assignments: dataset links

The mechanical engineering and agriculture assignment lists (`ml_mech_assignments.tex`, `ml_agri_assignments.tex`) use
datasets that are not stored here. Links only, no code; each slide already has the starter code.

Mechanical engineering

| Assignment | Dataset | Link |
|---|---|---|
| Predictive maintenance | NASA Turbofan Engine Degradation | https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository |
| Structural health monitoring | Search for a vibration or strain dataset | https://www.kaggle.com/datasets?search=structural+health+monitoring |
| Design optimization | DEAP library (no dataset, a problem to define) | https://deap.readthedocs.io/ |
| Fault detection in machinery | CWRU Bearing Data | https://engineering.case.edu/bearingdatacenter |
| Thermal analysis | Search for a thermal conductivity dataset | https://www.kaggle.com/datasets?search=thermal+conductivity |
| Quality control | Search for a manufacturing defect image dataset | https://www.kaggle.com/datasets?search=defect+detection |
| Energy consumption | Household Electric Power Consumption | https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption |
| Material properties | Materials Project | https://materialsproject.org/ |
| Strength prediction | Concrete Compressive Strength | https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength |
| Fluid dynamics | OpenFOAM simulations | https://www.openfoam.com/ |

Agriculture

| Assignment | Dataset | Link |
|---|---|---|
| Crop yield (linear regression) | Search for a crop yield dataset | https://www.kaggle.com/datasets?search=crop+yield+prediction |
| Crop disease (decision tree) | Plant Village | https://www.kaggle.com/datasets/emmarex/plantdisease |
| Weed detection (CNN) | Search for a weeds dataset | https://www.kaggle.com/datasets?search=weed+detection |
| Soil quality (random forest) | Search for a soil quality dataset | https://www.kaggle.com/datasets?search=soil+quality |
| Irrigation scheduling (SVM) | Search for an irrigation dataset | https://www.kaggle.com/datasets?search=irrigation+scheduling |
| Crop disease (KNN) | Search for a crop disease dataset | https://www.kaggle.com/datasets?search=crop+disease+prediction |
| Yield (gradient boosting) | USDA NASS crop data | https://www.nass.usda.gov/ |
| Pest detection (CNN) | Search for a pest image dataset | https://www.kaggle.com/datasets?search=pest+detection |
| Precision agriculture (K-Means) | Search for a precision agriculture dataset | https://www.kaggle.com/datasets?search=precision+agriculture |
| Weather forecasting (LSTM) | NOAA Climate Data Online | https://www.ncdc.noaa.gov/cdo-web/ |

Extra packages for the starter code on the slides, not part of `environment.yml`: `statsmodels` (the ARIMA energy assignment),
`deap` (the genetic algorithm assignment) and `keras` with `tensorflow` (the neural network, CNN and LSTM assignments). The
Keras snippets were run on Keras 3.7 and build without errors; the genetic algorithm snippet is an outline, not runnable as shown.

The Kaggle links are search pages and need a free Kaggle login to download. Kaggle blocks scripted access, so these links were
copied from the slides and not opened by a script; open them in a browser before class.
