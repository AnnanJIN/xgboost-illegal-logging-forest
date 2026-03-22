# ==============================================================================
# XGBoost Analysis of Forest Carbon Loss Drivers (2014-2019)
#
# Description:
#   This script trains an XGBoost model (Tweedie distribution) to identify
#   key drivers of forest carbon loss across Chinese cities, using Group
#   K-Fold cross-validation to prevent spatial data leakage.
#   SHAP values are computed for global and spatially stratified interpretation.
#
# Data:
#   The input dataset is available at: [INSERT YOUR REPOSITORY/DOI LINK HERE]
#   Download the CSV file and update `file_path` below before running.
#
# Required R packages (install if needed):
#   install.packages(c("tidyverse", "xgboost", "SHAPforxgboost",
#                      "caret", "corrplot", "car"))
#
# Tested on: R 4.3+
# ==============================================================================

# --- 0. Setup -----------------------------------------------------------------
rm(list = ls())
gc()

library(tidyverse)
library(xgboost)
library(SHAPforxgboost)
library(caret)
library(corrplot)
library(car)

# Set your working directory or update paths below
# All outputs will be saved to `out_dir`
out_dir <- "./xgboost_output/"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# --- 1. Data Loading and Preprocessing ----------------------------------------

# UPDATE THIS PATH to where you downloaded the dataset
file_path <- "./data/forest_carbon_loss_data.csv"

# Target variable: "sum_TCL" (total carbon loss) or "sum_XJL" (net carbon loss)
# Run the full script once for each target to reproduce both sets of results.
target_var <- "sum_XJL"

raw_data <- read.csv(file_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8")

df_model <- raw_data %>%
  # Restrict to study period
  filter(year >= 2014 & year <= 2019) %>%

  # Treat missing target values as zero (zero-inflated assumption)
  mutate(!!sym(target_var) := replace_na(!!sym(target_var), 0)) %>%

  # Log-transform right-skewed variables
  mutate(
    ln_GDP_PC     = log(gdp_per_cap + 1),
    ln_NightLight = log(NightLight_Mean + 1),
    ln_Freight    = log(GLHYL + 1),
    ln_Gov_Cap    = log(GGZLNL + 1),
    Year          = as.numeric(year)
  ) %>%

  # Impute remaining NAs with column median (maximises sample retention)
  mutate(across(
    c(forest_coverTDMJ, NPP_Mean, ln_Freight,
      DLMD, YCZB, ln_GDP_PC, ln_NightLight, ln_Gov_Cap),
    ~ ifelse(is.na(.), median(., na.rm = TRUE), .)
  )) %>%

  # Select and rename final modelling variables
  select(
    City_ID    = `修正城市名2`,   # City identifier
    Year,
    Y_Target   = !!sym(target_var),

    # Natural resources
    Forest_Cov = forest_coverTDMJ,
    NPP        = NPP_Mean,

    # Accessibility
    ln_Freight,
    Road_Den   = DLMD,

    # Economic incentives
    Prim_Ind      = YCZB,
    ln_GDP_PC,
    ln_NightLight,

    # Institutional capacity
    ln_Gov_Cap
  ) %>%
  na.omit()

cat("--- Sample summary ---\n")
cat("Observations:", nrow(df_model), "\n")
cat("Cities:", n_distinct(df_model$City_ID), "\n")
cat("Zero share of Y:", mean(df_model$Y_Target == 0), "\n")

write.csv(df_model,
          paste0(out_dir, "cleaned_modelling_data_", target_var, ".csv"),
          row.names = FALSE)

# --- 2. Collinearity Checks ---------------------------------------------------

# 2.1 Spearman correlation matrix (predictors only)
X_only  <- df_model %>% select(-City_ID, -Year, -Y_Target)
cor_mat <- cor(X_only, method = "spearman")

pdf(paste0(out_dir, "correlation_matrix_", target_var, ".pdf"),
    width = 10, height = 10)
corrplot(cor_mat, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black",
         number.cex = 0.8, tl.cex = 0.9, diag = FALSE,
         title = paste0("Spearman Correlation — ", target_var),
         mar = c(0, 0, 2, 0))
dev.off()

# 2.2 Variance Inflation Factor (VIF); values < 10 indicate acceptable collinearity
df_vif    <- df_model %>% select(-City_ID, -Year)
lm_dummy  <- lm(Y_Target ~ ., data = df_vif)
vif_vals  <- vif(lm_dummy)

cat("\n--- VIF results (threshold: 10) ---\n")
print(vif_vals)
write.csv(as.data.frame(vif_vals),
          paste0(out_dir, "vif_results_", target_var, ".csv"))

# --- 3. Group K-Fold Cross-Validation -----------------------------------------
# Cities are grouped into folds to prevent spatial leakage between train/test sets.

X_matrix <- as.matrix(df_model %>% select(-City_ID, -Year, -Y_Target))
Y_vector <- df_model$Y_Target

set.seed(2025)
city_list     <- unique(df_model$City_ID)
folds_indices <- createFolds(city_list, k = 5, list = TRUE, returnTrain = FALSE)

folds_list <- lapply(folds_indices, function(idx) {
  which(df_model$City_ID %in% city_list[idx])
})

# XGBoost hyperparameters — Tweedie objective suits zero-inflated count-like data
xgb_params <- list(
  objective             = "reg:tweedie",
  tweedie_variance_power = 1.5,
  eval_metric           = "rmse",
  eta                   = 0.05,
  max_depth             = 4,
  subsample             = 0.7,
  colsample_bytree      = 0.8,
  nthread               = 4
)

dtrain <- xgb.DMatrix(data = X_matrix, label = Y_vector)

cat("\nRunning 5-fold cross-validation...\n")
cv_results <- xgb.cv(
  params               = xgb_params,
  data                 = dtrain,
  nrounds              = 1000,
  folds                = folds_list,
  early_stopping_rounds = 50,
  verbose              = 0
)

best_iter <- cv_results$best_iteration
min_rmse  <- min(cv_results$evaluation_log$test_rmse_mean)
cat("Best iteration:", best_iter, "\n")
cat("Best CV RMSE:  ", round(min_rmse, 4), "\n")

write.csv(cv_results$evaluation_log,
          paste0(out_dir, "cv_training_log_", target_var, ".csv"))

# --- 4. Full-Data Model Training and SHAP Decomposition -----------------------

final_model <- xgb.train(
  params  = xgb_params,
  data    = dtrain,
  nrounds = best_iter,
  verbose = 0
)

xgb.save(final_model, paste0(out_dir, "xgboost_model_", target_var, ".model"))

cat("\nComputing SHAP values...\n")
shap_long <- shap.prep(xgb_model = final_model, X_train = X_matrix)

# Attach city/year labels for downstream stratification
shap_long$City_ID <- rep(df_model$City_ID, times = ncol(X_matrix))
shap_long$Year    <- rep(df_model$Year,    times = ncol(X_matrix))

# Global feature importance (mean absolute SHAP)
shap_importance <- shap_long %>%
  group_by(variable) %>%
  summarise(mean_shap = mean(abs(value))) %>%
  arrange(desc(mean_shap))

cat("\n--- Global feature importance (Top 5) ---\n")
print(head(shap_importance, 5))

write.csv(shap_long,
          paste0(out_dir, "shap_long_", target_var, ".csv"),
          row.names = FALSE)
write.csv(shap_importance,
          paste0(out_dir, "shap_importance_", target_var, ".csv"),
          row.names = FALSE)

# --- 5. Spatial Heterogeneity Analysis ----------------------------------------
# Cities are split into three ecological tiers based on mean NPP (30/70 percentiles).

city_endowment <- df_model %>%
  group_by(City_ID) %>%
  summarise(
    Mean_NPP     = mean(NPP,        na.rm = TRUE),
    Mean_Forest  = mean(Forest_Cov, na.rm = TRUE),
    Mean_Freight = mean(ln_Freight, na.rm = TRUE)
  ) %>%
  ungroup()

qt_high <- quantile(city_endowment$Mean_NPP, 0.70)
qt_low  <- quantile(city_endowment$Mean_NPP, 0.30)

label_high <- "High Ecological Value (Resource-Rich)"
label_mid  <- "Medium Ecological Value (Transition Zone)"
label_low  <- "Low Ecological Value (Resource-Poor)"

city_labels <- city_endowment %>%
  mutate(
    Region_Type = case_when(
      Mean_NPP >= qt_high ~ label_high,
      Mean_NPP <= qt_low  ~ label_low,
      TRUE                ~ label_mid
    ),
    Region_Type = factor(Region_Type,
                         levels = c(label_high, label_mid, label_low))
  )

cat("\n--- City group counts ---\n")
print(table(city_labels$Region_Type))

write.csv(city_labels,
          paste0(out_dir, "city_group_summary_", target_var, ".csv"),
          row.names = FALSE)

# SHAP importance by ecological tier (Top 8 drivers per group)
shap_spatial <- shap_long %>%
  left_join(city_labels %>% select(City_ID, Region_Type), by = "City_ID") %>%
  group_by(Region_Type, variable) %>%
  summarise(mean_shap = mean(abs(value)), .groups = "drop") %>%
  group_by(Region_Type) %>%
  slice_max(mean_shap, n = 8) %>%
  ungroup()

write.csv(shap_spatial,
          paste0(out_dir, "shap_spatial_heterogeneity_", target_var, ".csv"),
          row.names = FALSE)

cat("\n--- Spatial heterogeneity SHAP summary saved ---\n")
cat("All outputs written to:", out_dir, "\n")
