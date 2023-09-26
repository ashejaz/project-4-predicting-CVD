-- Drops the table named CVD
DROP TABLE CVD;

-- Creates a new table named CVD and defines the datatypes of each field
CREATE TABLE CVD (
	General_Health VARCHAR(20),
    Checkup VARCHAR(50),
    Exercise VARCHAR(3),
    Heart_Disease VARCHAR(3),
    Skin_Cancer VARCHAR(3),
    Other_Cancer VARCHAR(3),
    Depression VARCHAR(3),
    Diabetes VARCHAR(50),
    Arthritis VARCHAR(3),
    Sex VARCHAR(10),
    Age_Category VARCHAR(10),
    Height_cm DECIMAL,
    Weight_kg DECIMAL,
    BMI DECIMAL,
    Smoking_History VARCHAR(3),
    Alcohol_Consumption DECIMAL,
    Fruit_Consumption DECIMAL,
    Green_Vegetables_Consumption DECIMAL,
    FriedPotato_Consumption DECIMAL
);

-- The CVD_dataset_uncleaned.csv was imported into the CVD table using the postgres GUI 

-- The diabetes field contained values which were not beneficial to our analysis so they were deleted from the table.
DELETE FROM CVD 
WHERE diabetes = 'No, pre-diabetes or borderline diabetes' OR diabetes = 'Yes, but female told only during pregnancy';

-- Selects all the data in the cleaned CVD table and a copy of this output was exported and saved in the Resources folder as CVD_dataset_cleaned.csv 
SELECT * 
FROM CVD;