-- Example warehouse analysis for data/processed/featured_housing.csv.
SELECT ocean_proximity, COUNT(*) AS homes,
       AVG(median_house_value) AS average_value,
       AVG(median_income) AS average_income
FROM housing_features
GROUP BY 1
ORDER BY average_value DESC;
