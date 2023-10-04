# Cardiovascular Disease Predictor

For this group project, a series of supervised machine learning models were created and optimised to predict cardiovascular disease based on risk factors.

![internal_structures_1_0](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/3f8b5491-c280-4175-9cc8-f7b11872d568)

-----------------------
Team Members:
- Ash Ejaz
- Ayroza Dobson
- Lishani Srikaran
- Savina Boateng
-----------------------

## Data Source and Cleaning

Our Cardiovascular Risk Factor dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset), with the original source of the data being sourced from the 2021 [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/index.html) data collected by the Centres for Disease Control (CDC). The BRFSS is the largest continuously conducted health survey system in the world, collecting data from U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. 

The Kaggle dataset consists of a condensed version of the BRFSS, including whether an individual did or did not have heart disease as well as 18 risk factors for heart disease identified by the publisher. The dataset was downloaded in [CSV format](Resources/CVD_dataset_uncleaned.csv) and loaded into a PostgreSQL database where it was cleaned. The script for this step can be found [here](SQL/CVD_table_cleaned.sql).

The cleaned data was then uploaded to a public [Google Sheets link](https://docs.google.com/spreadsheets/d/e/2PACX-1vSDchXr1EhgCSsxlxJ3lWPhh1kT5EJS3yv4DJ2YLeMIC3y4uq-Pp4EQknrs9zAiaI3ulne2Jyi6gR6G/pub?gid=602879552&single=true&output=csv) for easier access when using Google Colab.

Additional insights were also created in a separate Tableau workbook to gain a wider bredth of knowledge in relation to ethnicity and territory, this data was sourced from [CDC](https://data.cdc.gov/Heart-Disease-Stroke-Prevention/Heart-Disease-Mortality-Data-Among-US-Adults-35-by/jiwm-ppbh). This dataset contained heart disease mortality data among US adults (35+) by state/ territory and country - 2018-2020. 

## Data Exploration

The features within the dataset were explored and visualised using PySpark, SparkSQL and Plotly.

For each risk factor, the following visualisations were created:

1) The breakdown of the risk factor demographic within the dataset was explored.
2) The proportion of each risk factor present in heart disease cases.
3) The prevalence per each risk factor in heart disease cases to remove biases of group ratios. 

For example, for the 'exercise' risk factor we were able to visualise the following:

1) 
![image](https://github.com/ashejaz/project-4-predicting-CVD/assets/126973634/af62a956-ed5f-4bc7-9aa3-f7db8c52803f)

2)
![image](https://github.com/ashejaz/project-4-predicting-CVD/assets/126973634/783ad295-4243-4ed1-9b2e-52e4e3df0438)

3) 
![image](https://github.com/ashejaz/project-4-predicting-CVD/assets/126973634/912117f1-0365-4c3d-bae7-34c4221accae)


The script containing all script and visualisations can be found [here](Notebooks/1-Data_Exploration/cvd_data_exploration.ipynb).

Furthermore, a [Tableau dashboard](https://public.tableau.com/app/profile/ayroza.dobson/viz/Project4-PredictingCVD/Story1?publish=yes) was created to convey the disparity of heart disease mortality by state, gender, and race to an audience.

## Data Preprocessing
- The classes in our dataset were heavily imbalanced, for every 1 person who had heart disease, there were 12 who did not.Therefore all datapoints outside of 1 standard deviation from the mean were removed.   This, along with random oversampling allowed us to balance the classes in our dataset.
- Numerical columns were scaled with StandardScalar and all categorical columns were encoded using get_dummies. 
- The preprocessed data was split into a testing and training set.

The script containing the full preprocessing steps as well as the 5 initial models can be found [here](Notebooks/2-Prediction_Models/cvd_prediction_models.ipynb).

## Model 1: Logistic Regression

Logistic regression is a statistical method for binary classification, such as predicting whether someone has Cardiovascular Disease (CVD) or not based on multiple factors like age, BMI, and general health. It employs a sigmoid function to transform the combined effects of these factors into a probability of CVD presence. By estimating coefficients through several techiques, it establishes a decision boundary and classifies individuals as either having CVD or not. Model performance is assessed using metrics like accuracy and precision.

The solver we used to optimize the parameters of the logistic regression model was 'saga' which os better suited for datasets with a large number of features.

The following classification report was produced:

![Screenshot 2023-10-04 at 19 32 21](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/7e495c73-30e3-46f1-9b6f-cffcfb4eb328)

Hyperparameter tuning was explored to enhance the logistic regression model's performance. Initially, the 'saga' solver was chosen for its effectiveness with large datasets, and alternatives like 'lbfgs' weren't suitable for our specific problem. 'sag' was not used because it's an older version of 'saga' and 'saga' remained the preferred choice. Hence, optimisation was not performed for this model.

## Model 2: Support Vector Machines

Support Vector Machines (SVMs) are classification algorithms that find a hyperplane, or decision boundary, in multi-dimensional spaces to separate two classes, like CVD and non-CVD. They excel by maximizing the margin, which is the space between the boundary and the nearest data points, making them robust for complex datasets. SVMs can adapt to non-linear data through the kernel trick, handle multi-class classification, and offer control through a regularisation parameter.

The following classification report was produced:

![image (1)](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/c2ab1114-c29e-4a17-93b7-3f46ac5bb9dc)

### Optimisation

The kernel hyperparameters of this model were adjusted to potentially better suit our dataset's structure, which gave the following results:

![image (2)](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/3bffba86-706c-458d-88d0-14c51005156e)

The full model optimisation and analysis of results can be found [here](Notebooks/3-Optimised_Prediction_Models/SVM_model_optimisation.ipynb).

## Model 3: Decision Tree

Decision Trees are hierarchical, non-linear structures that recursively divide a dataset into subsets by selecting the most informative features at each node. This selection is based on criteria like Gini impurity or information gain, which measure the data's homogeneity within each branch. Decision Trees create a series of if-then-else conditions that navigate from the root node to leaf nodes, where decisions are made. This tree structure offers interpretability, as you can trace the path from root to leaf to understand how a decision is reached.

The following classification report was produced:

![Screenshot 2023-10-04 at 20 41 55](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/8a5f40c0-6b14-4025-a9c8-45d79c474bb3)

### Optimisation

The optimisation of the Decision Tree model focused on manually fine-tuning hyperparameters such as max_depth, min_samples_split, and min_samples_leaf to specific values. By limiting the tree's depth and controlling the granularity of splits, we aimed to prevent overfitting and ensure the model's generalisation to new data.

This yielded the following results:

![Screenshot 2023-10-04 at 20 44 33](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/d2e4b72e-db9c-4fdc-919a-0546e929f21d)

The full model optimisation and analysis of results can be found [here](Notebooks/3-Optimised_Prediction_Models/decision_tree_optimisation.ipynb).

## Model 4: Random Forest

Random Forests are ensemble methods that enhance predictive accuracy and alleviate overfitting by aggregating the outputs of multiple Decision Trees. They work by creating a collection of Decision Trees, each trained on a different subset of the data and employing random feature subsets. These trees' results are combined, reducing variance and improving generalization. Additionally, Random Forests offer feature importance analysis, measuring how much each feature influences predictions.

The following classification report was produced:

![Screenshot 2023-10-04 at 20 47 37](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/5366a0de-859d-430e-a8bb-ca34eec9ae0c)


### Optimisation

The optimisation of the Random Forest model involved defining a parameter distribution dictionary for key hyperparameters with a RandomizedSearchCV object named random_search, employing cross-validation and parallel processing to find the best hyperparameter combination for improving model accuracy. 

This yielded the following results:

![Screenshot 2023-10-04 at 20 51 08](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/334c4256-a4af-4b7f-8c18-a0d1e1f19e0f)

The full model optimisation and analysis of results can be found [here](Notebooks/3-Optimised_Prediction_Models/random_forest_optimisation.ipynb).

## Model 5: Neural Network

Neural Networks have gained prominence owing to their capacity to decipher intricate data patterns. These models comprise layers of interconnected artificial neurons, each performing weighted computations. They excel in feature extraction and nonlinear modeling, allowing them to capture intricate relationships between predictors and CVD outcomes. Neural Networks are particularly effective when ample data is available, as they can learn complex, data-driven representations that may elude more traditional models.

An initial model with 2 hidden layers and 64 nodes in each layer produced the following classification report:

![Screenshot 2023-10-04 at 21 19 53](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/a1ddc069-af32-4493-be79-aaa40f10159d)

However, after around 20 epochs the validation accuracy plateaud and started to decrease, and the validation loss appeared to increase. This suggested that the model was overfitting and that though it performs well on the training data, it may struggle on unseen data.

### Optimisation 1

The kerastuner was used to select the best hyperparameters for the model which selected 5 layers. Early stopping was used to reduce overfitting.

The following classification report was produced:

![Screenshot 2023-10-04 at 21 43 11](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/347bd945-2e9f-4b97-8ef4-02ba98edb8a4)

However, though the model performance improved, overfitting was still occuring.

The full model optimisation and analysis of results can be found [here](Notebooks/3-Optimised_Prediction_Models/neural_network_optimisation_1.ipynb).

### Optimisation 2

The hyperparameters for the 3rd best performing model were selected as they involved only 2 layers with similar accuracy values to the best performing model. 

The hope was that by reducing the complexity of the model, overfitting would be reduced and the accuracy would be maintained.

The classification report however, did not show significant improvement compared to the previous optimisation:

![Screenshot 2023-10-04 at 21 51 10](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/05027232-fa8a-496b-84e6-2e5c2baa5fb4)

This suggests that overfitting is an ongoing issue with our data.

The full model optimisation and analysis of results can be found [here](Notebooks/3-Optimised_Prediction_Models/neural_network_optimisation_2.ipynb).

H5 files for all three neural network models ran can be found in the [h5 folder](h5).

## Summary

Initially, the Random Forest Model performed the best out of the 5 models, reaching 93% for both accuracy and precision. 

**ADD CHART HERE**

However, after optimisation, the Decision Tree became the best performing model achieving 93% accuracy and 94% precision beating the Random Forest by 1% precision.

**ADD CHART HERE**

The script and analysis for the comparison of our optimised models can be found [here](Notebooks/4-Best_Model/cvd_best_model.ipynb).

A [summary table](model_performance_summary.xlsx) with accuracy and precision values for all models is pictured below:

![Screenshot 2023-10-04 at 18 13 09](https://github.com/ashejaz/project-4-predicting-CVD/assets/127614970/c96e149a-24a9-42ca-9958-c9ba3e739563)

## Limitations and Acknowledgements

- Overfitting persisted despite implementation of early stopping and hyperparameter tuning, evident by the higher training accuracy compared to validation accuracy.
- Removal of outliers from the majority class resulted in a significant reduction in the size of the dataset which may have led to loss of valuable information.
- Our dataset is geographically limited to the United States which restricts its global applicability.

## Conclusions

We recommend the Random Forest model to be considered for further exploration due to its nature of ensemble learning and its consistently high precision and accuracy scores.

We believe the model requires significant further optimisation before deployment due to the sensitivity of the topic and the fact that false positive and false negative predictions should be reduced as much as possible.

Our next steps would be the following:

- Explore advanced regularisation techniques to further combat overfitting
- Seek methods to balance class distribution without sacrificing dataset richness
- Expand our data sources beyond the United States
- Place increased emphasis on feature engineering and ongoing model evaluation to enhance the predictive capabilities

## Risk Factor Resource Webpage

The best means of prevention is through education which is why we created a [web page](https://ashejaz.github.io/project-4-predicting-CVD/) containing links to resources related to each cardiovascular disease risk indicator. 

For example, for someone in the US who doesn't exercise and wanted to find a class, they could visit our page and find a link which will help them to find a local class.



