# Model card

The model estimates median house value from geography, age, rooms, bedrooms, population, households, income, ocean proximity, and three derived ratios. Ridge is the linear baseline; histogram gradient boosting is the nonlinear candidate. The held-out RMSE determines which pipeline is served.

Limits: the fallback is synthetic; results omit market time, transactions, school quality, zoning, condition, protected attributes, and geographic fairness analysis. Predictions are estimates for demonstration, not appraisals or financial decisions.
