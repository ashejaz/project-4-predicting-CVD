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

## Model 1: Logistic Regression

## Model 2: Support Vector Machines

## Model 3: Decision Tree

## Model 4: Random Forest

## Model 5: Neural Network

## Summary

Initially, the Random Forest Model performed the best out of the 5 models, reaching 93% for both accuracy and precision. 

However, after optimisation, the Decision Tree became the best performing model achieving 93% accuracy and 94% precision beating the Random Forest by 1% precision.

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



