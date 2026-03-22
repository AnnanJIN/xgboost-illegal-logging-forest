# xgboost-illegal-logging-forest
We apply an **XGBoost model with Tweedie objective** to quantify and interpret the drivers of illegal timber harvesting volume across Chinese prefecture-level cities from 2014 to 2019. 
Drivers of Illegal logging in China: A City-Level XGBoost–SHAP Analysis (January 2014 - December 2019)
Overview
We apply an XGBoost model (Tweedie objective) to identify the socioeconomic and ecological drivers of illegal timber harvesting volume (sum_XJL,) across Chinese prefecture-level cities from 2014 to 2019. SHAP values are used to interpret driver contributions at global scale and across cities stratified by ecological endowment. Full variable definitions and model specifications are provided in the paper.

Requirements
R ≥ 4.3
install.packages(c(
  "tidyverse",
  "xgboost",
  "SHAPforxgboost",
  "caret",
  "corrplot",
  "car"
))

Data Availability
  1     The original dataset is not publicly available. 
  2     Access can be requested by contacting Gang LI (lig@nwu.edu.cn).
Usage
	1	The origenally dataset should be apply to Gang LI (lig@nwu.edu.cn), you can read the data irodaction documents in Zenodo (DOI:)
	2	Open xgboost_forest_loss_analysis.R and run the full script
	3	Outputs are saved automatically to xgboost_output/

Repository Structure
├── xgboost_forest_loss_analysis.R   # Main analysis script
├── data/
│   └── illegal_logging_forest.csv   # need apply 
├── xgboost_output/                  # Generated on first run
├── .zenodo.json
└── README.md


License
Code: MIT License
