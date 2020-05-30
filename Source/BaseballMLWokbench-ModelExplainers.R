# TESTED ON CRAN R 4.0.0
# Last Updated: 05/29/2020

# Check your R environment version
# Should be close to the tested version(s)
version

# 1) INSTALL & LOAD PACKAGES
install.packages("devtools")

# a) INSTALL Core Modeling Packages
install.packages("corrplot")
install.packages('Hmisc')
# b) ML Orchestration/Explainer/Helper Packages
install.packages('auditor')
install.packages('ceterisParibus')
install.packages('breakDown')
install.packages('DALEX')
install.packages('DALEXtra')
install.packages('ingredients')
install.packages('jsonlite')
install.packages('purrr')

# b) LOAD Packages
#Core Statistical Packages
library(auditor)
library(corrplot)
library(ggplot2)
library(reshape2)
library(Hmisc)
library(httr)
library(purrr)
library(stats)
# ML Orchestration/Explainer/Helper Packages
library(ceterisParibus)     # Used for What-If Charts
library(breakDown)          # Used for Model Explainers
library(ingredients)        # Used for Model Explainers
library(DALEX)              # Used for Model Explainers
library(DALEXtra)           # Used for Model Explainers
library(jsonlite)           # Used for Model Explainers
library(pdp)                # Used for Model Explainers

# c) Check versions (optional)
sessionInfo()

# 2) SETUP TRAINING, VALIDATION & COMBINED DATA SETS

# a) Set working directory
getwd()
# Working Directory format for macOS
# wdPath <- "/Users/bartczernicki-msft/Desktop/SourceCode/BaseballHOFDalex"
# Working Directory format for Windows OS
wdPath <- "C:\\Users\\Bart\\source\\repos\\BaseballHOFPredictionWithMlrAndDALEX-master\\Source"
PLOTPATHS <- "C:\\Users\\Bart\\source\\repos\\BaseballHOFPredictionWithMlrAndDALEX-master\\Images\\BaseballMLWorkbench\\"
setwd(wdPath)

# b) Load Training Data CSV File
MLBBaseballBatters <- read.csv(file="MLBBaseballBatters.csv", header=TRUE, sep=",")

# c) Change InductedToHallOfFame & OnHallOfFameBallot labels as a 1/0 binary factor
MLBBaseballBatters$InductedToHallOfFame <- factor(ifelse(MLBBaseballBatters$InductedToHallOfFame =='TRUE', 1, 0))
MLBBaseballBatters$OnHallOfFameBallot <- factor(ifelse(MLBBaseballBatters$OnHallOfFameBallot =='TRUE', 1, 0))

# d) Remove ID columns (LastYearPlayed, ID) not needed for training classifier models
y_OnHallOfFameBallot <- as.numeric(as.character(MLBBaseballBatters$OnHallOfFameBallot))
y_InductedToHallOfFame <- as.numeric(as.character(MLBBaseballBatters$InductedToHallOfFame))
MLBBaseballBatters <- subset(MLBBaseballBatters, select = -c(LastYearPlayed, ID, OnHallOfFameBallot, InductedToHallOfFame))
nrow(MLBBaseballBatters)
azureFunctionBaseUrl <- "http://localhost:7071/api/MakeBaseballPrediction"

getAzureFunctionUrl <- function(dev, azureFunctionBaseUrl, predictionType, modelAlgorithm)
{
  functionUrl <- ""
  if(isTRUE(dev))
  {
    functionUrl <- paste(azureFunctionBaseUrl, "?PredictionType=", predictionType, "&ModelAlgorithm=", modelAlgorithm, sep="")
  } else
  {
    functionUrl <- paste(azureFunctionBaseUrl, "&PredictionType=", predictionType, "&ModelAlgorithm=", modelAlgorithm, sep="")
  }
  
  return (functionUrl)
}

savePlot <- function(plotName, plotObject, showPlot)
{
  if(isTRUE(showPlot))
  {
    plot(plotObject)
  }
  
  png(paste(PLOTPATHS, plotName, ".png",sep=""))
  plot(plotObject)
  dev.off()
  
  return(0)
}

# f) Check Validation Data
head(MLBBaseballBatters, 15)

#############################################
# 3) MACHINE LEARNING - GLOBAL MODEL EXPLAINER


# a) Build a POST HTTP request function to extract prediction data, required for DALEX explainers
# Test
# getAzureFunctionUrl(azureFunctionBaseUrl, "OnHallOfFameBallot", "LightGbm")
# top5RowsJSON <- toJSON(head(MLBBaseballBatters, 5), pretty=TRUE, auto_unbox=TRUE)

# Build Function that executes the Predictions Call
getBaseballHofPrediction <- function(httpObject, newdata)
{
  jsonRows <- toJSON(newdata, pretty=TRUE, auto_unbox=TRUE)
  url <- getAzureFunctionUrl(httpObject$IsDevelopment, httpObject$AzureFunctionBaseUrl, httpObject$PredictionType, httpObject$ModelAlgorithm)
  requestPOST <- POST(url, body = jsonRows, encode = "json")
  probabilities <- fromJSON(content(requestPOST))
  return(probabilities)
}


# b) Build model explainers for each of the used algorithms
# ?DALEX::explain
IsDevelopment <- TRUE
#FastTree
httpObject_FastTreeOnHallOfFameBallot <-list(IsDevelopment=TRUE, PredictionType="OnHallOfFameBallot", ModelAlgorithm="FastTree", AzureFunctionBaseUrl=azureFunctionBaseUrl)
explainer_RemoteMlNet_FastTreeOnHallOfFameBallot <- DALEX::explain(httpObject_FastTreeOnHallOfFameBallot, data=MLBBaseballBatters, y=y_OnHallOfFameBallot, label= "MLNetFastTree-OnHallOfFameBallot", predict_function = getBaseballHofPrediction, type="classification")
httpObject_FastTreeInductedToHallOfFame <-list(IsDevelopment=TRUE, PredictionType="InductedToHallOfFame", ModelAlgorithm="FastTree", AzureFunctionBaseUrl=azureFunctionBaseUrl)
explainer_RemoteMlNet_FastTreeInductedToHallOfFame <- DALEX::explain(httpObject_FastTreeInductedToHallOfFame, data=MLBBaseballBatters, y=y_InductedToHallOfFame, label= "MLNetFastTree-InductedToHallOfFame", predict_function = getBaseballHofPrediction, type="classification")
#GeneralizedAdditiveModels
httpObject_GeneralizedAdditiveModelsOnHallOfFameBallot <-list(IsDevelopment=TRUE, PredictionType="OnHallOfFameBallot", ModelAlgorithm="GeneralizedAdditiveModels", AzureFunctionBaseUrl=azureFunctionBaseUrl)
explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot <- DALEX::explain(httpObject_GeneralizedAdditiveModelsOnHallOfFameBallot, data=MLBBaseballBatters, y=y_OnHallOfFameBallot, label= "MLNetGeneralizedAdditiveModels-OnHallOfFameBallot", predict_function = getBaseballHofPrediction, type="classification")
httpObject_GeneralizedAdditiveModelsInductedToHallOfFame <-list(IsDevelopment=TRUE, PredictionType="InductedToHallOfFame", ModelAlgorithm="GeneralizedAdditiveModels", AzureFunctionBaseUrl=azureFunctionBaseUrl)
explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame <- DALEX::explain(httpObject_GeneralizedAdditiveModelsInductedToHallOfFame, data=MLBBaseballBatters, y=y_InductedToHallOfFame, label= "MLNetGeneralizedAdditiveModels-InductedToHallOfFame", predict_function = getBaseballHofPrediction, type="classification")
#LightGbm
httpObject_LightGbmOnHallOfFameBallot <-list(IsDevelopment=TRUE, PredictionType="OnHallOfFameBallot", ModelAlgorithm="LightGbm", AzureFunctionBaseUrl=azureFunctionBaseUrl)
explainer_RemoteMlNet_LightGbmOnHallOfFameBallot <- DALEX::explain(httpObject_LightGbmOnHallOfFameBallot, data=MLBBaseballBatters, y=y_OnHallOfFameBallot, label= "MLNetLightGbm-OnHallOfFameBallot", predict_function = getBaseballHofPrediction, type="classification")
httpObject_LightGbmInductedToHallOfFame <-list(IsDevelopment=TRUE, PredictionType="InductedToHallOfFame", ModelAlgorithm="LightGbm", AzureFunctionBaseUrl=azureFunctionBaseUrl)
explainer_RemoteMlNet_LightGbmInductedToHallOfFame <- DALEX::explain(httpObject_LightGbmInductedToHallOfFame, data=MLBBaseballBatters, y=y_InductedToHallOfFame, label= "MLNetLightGbm-InductedToHallOfFame", predict_function = getBaseballHofPrediction, type="classification")


# c) Global Explainer - Model Performance
# ?model_performance
#FastTree
modelPerformance_RemoteMlNetGam_FastTreeInductedToHallOfFame <- DALEX::model_performance(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, cutoff = 0.5)
modelPerformance_RemoteMlNetGam_FastTreeOnHallOfFameBallot <- DALEX::model_performance(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, cutoff = 0.5)
#GeneralizedAdditiveModels
modelPerformance_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame <- DALEX::model_performance(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, cutoff = 0.5)
modelPerformance_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot <- DALEX::model_performance(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, cutoff = 0.5)
#LightGbm
modelPerformance_RemoteMlNetGam_LightGbmInductedToHallOfFame <- DALEX::model_performance(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, cutoff = 0.5)
modelPerformance_RemoteMlNetGam_LightGbmOnHallOfFameBallot <- DALEX::model_performance(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, cutoff = 0.5)
plotModelPerformanceInductedToHallOfFame <- plot(modelPerformance_RemoteMlNetGam_FastTreeInductedToHallOfFame, modelPerformance_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame, modelPerformance_RemoteMlNetGam_LightGbmInductedToHallOfFame)
savePlot("GlobalExplainer-ModelPerformance-Models-InductedToHallOfFame", plotModelPerformanceInductedToHallOfFame, TRUE)
plotModelPerformanceOnHallOfFameBallot <- plot(modelPerformance_RemoteMlNetGam_LightGbmOnHallOfFameBallot, modelPerformance_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot, modelPerformance_RemoteMlNetGam_LightGbmOnHallOfFameBallot)
savePlot("GlobalExplainer-ModelPerformance-Models-OnHallOfFameBallot", plotModelPerformanceOnHallOfFameBallot, TRUE)

# d) Global Explainer - Model Parts | Feature Importance 
?ingredients::feature_importance
#FastTree
featureImportance_RemoteMlNetGam_FastTreeInductedToHallOfFame <- ingredients::feature_importance(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, loss_function = loss_root_mean_square)
featureImportance_RemoteMlNetGam_FastTreeOnHallOfFameBallot <- ingredients::feature_importance(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, loss_function = loss_root_mean_square)
#GeneralizedAdditiveModels
featureImportance_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame <- feature_importance(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, loss_function = loss_root_mean_square)
featureImportance_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot <- feature_importance(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, loss_function = loss_root_mean_square)
#LightGbm
featureImportance_RemoteMlNetGam_LightGbmInductedToHallOfFame <- feature_importance(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, loss_function = loss_root_mean_square)
featureImportance_RemoteMlNetGam_LightGbmOnHallOfFameBallot <- feature_importance(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, loss_function = loss_root_mean_square)
savePlot("GlobalExplainer-ModelParts-FeatureImportance-InductedToHallOfFame-FastTree", plot(featureImportance_RemoteMlNetGam_FastTreeInductedToHallOfFame), TRUE)
savePlot("GlobalExplainer-ModelParts-FeatureImportance-InductedToHallOfFame-GeneralizedAdditiveModels", plot(featureImportance_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame), TRUE)
savePlot("GlobalExplainer-ModelParts-FeatureImportance-InductedToHallOfFame-LightGbm", plot(featureImportance_RemoteMlNetGam_LightGbmInductedToHallOfFame), TRUE)
savePlot("GlobalExplainer-ModelParts-FeatureImportance-OnHallOfFameBallot-FastTree", plot(featureImportance_RemoteMlNetGam_FastTreeOnHallOfFameBallot), TRUE)
savePlot("GlobalExplainer-ModelParts-FeatureImportance-OnHallOfFameBallot-GeneralizedAdditiveModels", plot(featureImportance_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot), TRUE)
savePlot("GlobalExplainer-ModelParts-FeatureImportance-OnHallOfFameBallot-LightGbm", plot(featureImportance_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot), TRUE)


# e) Global Explainer - Model Profile | Partial Dependency Profile
# ?model_profile
#FastTree -TotalPlayerAwards
modelProfile_RemoteMlNetGam_FastTreeInductedToHallOfFame_TotalPlayerAwards <- model_profile(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, variables = "TotalPlayerAwards", N=200)
modelProfile_RemoteMlNetGam_FastTreeOnHallOfFameBallot_TotalPlayerAwards <- model_profile(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, variables = "TotalPlayerAwards", N=200)
#GeneralizedAdditiveModels -TotalPlayerAwards
modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_TotalPlayerAwards <- model_profile(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, variables = "TotalPlayerAwards", N=200)
modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_TotalPlayerAwards <- model_profile(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, variables = "TotalPlayerAwards", N=200)
#LightGbm -TotalPlayerAwards
modelProfile_RemoteMlNetGam_LightGbmInductedToHallOfFame_TotalPlayerAwards <- model_profile(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, variables = "TotalPlayerAwards", N=200)
modelProfile_RemoteMlNetGam_LightGbmOnHallOfFameBallot_TotalPlayerAwards <- model_profile(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, variables = "TotalPlayerAwards", N=200)

savePlot("GlobalExplainer-ModelProfile-PartialDependencyProfile-OnHallOfFameBallot-TotalPlayerAwards-Models", plot(modelProfile_RemoteMlNetGam_FastTreeOnHallOfFameBallot_TotalPlayerAwards$agr_profiles,
                                                                                                 modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_TotalPlayerAwards$agr_profiles,
                                                                                                 modelProfile_RemoteMlNetGam_LightGbmOnHallOfFameBallot_TotalPlayerAwards$agr_profiles), TRUE)
savePlot("GlobalExplainer-ModelProfile-PartialDependencyProfile-InductedToHallOfFame-TotalPlayerAwards-Models", plot(modelProfile_RemoteMlNetGam_FastTreeInductedToHallOfFame_TotalPlayerAwards$agr_profiles,
                                                                                                 modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_TotalPlayerAwards$agr_profiles,
                                                                                                 modelProfile_RemoteMlNetGam_LightGbmInductedToHallOfFame_TotalPlayerAwards$agr_profiles), TRUE)

#FastTree - HR
modelProfile_RemoteMlNetGam_FastTreeInductedToHallOfFame_HR <- model_profile(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, variables = "HR", N=200)
modelProfile_RemoteMlNetGam_FastTreeOnHallOfFameBallot_HR <- model_profile(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, variables = "HR", N=200)
#GeneralizedAdditiveModels -HR
modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_HR <- model_profile(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, variables = "HR", N=200)
modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_HR <- model_profile(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, variables = "HR", N=200)
#LightGbm -HR
modelProfile_RemoteMlNetGam_LightGbmInductedToHallOfFame_HR <- model_profile(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, variables = "HR", N=200)
modelProfile_RemoteMlNetGam_LightGbmOnHallOfFameBallot_HR <- model_profile(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, variables = "HR", N=200)

savePlot("GlobalExplainer-ModelProfile-PartialDependencyProfile-OnHallOfFameBallot-HR-Models", plot(modelProfile_RemoteMlNetGam_FastTreeOnHallOfFameBallot_HR$agr_profiles,
                                                                                                    modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_HR$agr_profiles,
                                                                                                    modelProfile_RemoteMlNetGam_LightGbmOnHallOfFameBallot_HR$agr_profiles), TRUE)
savePlot("GlobalExplainer-ModelProfile-PartialDependencyProfile-InductedToHallOfFame-HR-Models", plot(modelProfile_RemoteMlNetGam_FastTreeInductedToHallOfFame_HR$agr_profiles,
                                                                                                      modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_HR$agr_profiles,
                                                                                                      modelProfile_RemoteMlNetGam_LightGbmInductedToHallOfFame_HR$agr_profiles), TRUE)

#FastTree - AB
modelProfile_RemoteMlNetGam_FastTreeInductedToHallOfFame_AB <- model_profile(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, variables = "AB", N=200)
modelProfile_RemoteMlNetGam_FastTreeOnHallOfFameBallot_AB <- model_profile(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, variables = "AB", N=200)
#GeneralizedAdditiveModels -AB
modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_AB <- model_profile(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, variables = "AB", N=200)
modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_AB <- model_profile(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, variables = "AB", N=200)
#LightGbm -AB
modelProfile_RemoteMlNetGam_LightGbmInductedToHallOfFame_AB <- model_profile(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, variables = "AB", N=200)
modelProfile_RemoteMlNetGam_LightGbmOnHallOfFameBallot_AB <- model_profile(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, variables = "AB", N=200)

savePlot("GlobalExplainer-ModelProfile-PartialDependencyProfile-OnHallOfFameBallot-AB-Models", plot(modelProfile_RemoteMlNetGam_FastTreeOnHallOfFameBallot_AB$agr_profiles,
                                                                                                    modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_AB$agr_profiles,
                                                                                                    modelProfile_RemoteMlNetGam_LightGbmOnHallOfFameBallot_AB$agr_profiles), TRUE)
savePlot("GlobalExplainer-ModelProfile-PartialDependencyProfile-InductedToHallOfFame-AB-Models", plot(modelProfile_RemoteMlNetGam_FastTreeInductedToHallOfFame_AB$agr_profiles,
                                                                                                      modelProfile_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_AB$agr_profiles,
                                                                                                      modelProfile_RemoteMlNetGam_LightGbmInductedToHallOfFame_AB$agr_profiles), TRUE)


# f) Global Explainer - Model Diagnostics - Variable Effect Partial Dependecy
?variable_effect_partial_dependency

#Feature - TotalPlayerAwards
#FastTree
varEffectPartialDependency_RemoteMlNetGam_FastTreeInductedToHallOfFame_TotalPlayerAwards  <- variable_effect_partial_dependency(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, variable = "TotalPlayerAwards")
varEffectPartialDependency_RemoteMlNetGam_FastTreeOnHallOfFameBallot_TotalPlayerAwards  <- variable_effect_partial_dependency(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, variable = "TotalPlayerAwards")
#GeneralizedAdditiveModels
varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_TotalPlayerAwards  <- variable_effect_partial_dependency(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, variable = "TotalPlayerAwards")
varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_TotalPlayerAwards  <- variable_effect_partial_dependency(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, variable = "TotalPlayerAwards")
#LightGbm
varEffectPartialDependency_RemoteMlNetGam_LightGbmInductedToHallOfFame_TotalPlayerAwards  <- variable_effect_partial_dependency(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, variable = "TotalPlayerAwards")
varEffectPartialDependency_RemoteMlNetGam_LightGbmOnHallOfFameBallot_TotalPlayerAwards  <- variable_effect_partial_dependency(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, variable = "TotalPlayerAwards")
#Plots
plot(varEffectPartialDependency_RemoteMlNetGam_FastTreeInductedToHallOfFame_TotalPlayerAwards,varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_TotalPlayerAwards,varEffectPartialDependency_RemoteMlNetGam_LightGbmInductedToHallOfFame_TotalPlayerAwards)
plot(varEffectPartialDependency_RemoteMlNetGam_FastTreeOnHallOfFameBallot_TotalPlayerAwards, varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_TotalPlayerAwards, varEffectPartialDependency_RemoteMlNetGam_LightGbmOnHallOfFameBallot_TotalPlayerAwards)

#Feature - TB
#FastTree
varEffectPartialDependency_RemoteMlNetGam_FastTreeInductedToHallOfFame_TB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, variable = "TB")
varEffectPartialDependency_RemoteMlNetGam_FastTreeOnHallOfFameBallot_TB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, variable = "TB")
#GeneralizedAdditiveModels
varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_TB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, variable = "TB")
varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_TB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, variable = "TB")
#LightGbm
varEffectPartialDependency_RemoteMlNetGam_LightGbmInductedToHallOfFame_TB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, variable = "TB")
varEffectPartialDependency_RemoteMlNetGam_LightGbmOnHallOfFameBallot_TB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, variable = "TB")
#Plots
plot(varEffectPartialDependency_RemoteMlNetGam_FastTreeInductedToHallOfFame_TB,varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_TB,varEffectPartialDependency_RemoteMlNetGam_LightGbmInductedToHallOfFame_TB)
plot(varEffectPartialDependency_RemoteMlNetGam_FastTreeOnHallOfFameBallot_TB, varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_TB, varEffectPartialDependency_RemoteMlNetGam_LightGbmOnHallOfFameBallot_TB)

#Feature - SB
#FastTree
varEffectPartialDependency_RemoteMlNetGam_FastTreeInductedToHallOfFame_SB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, variable = "SB")
varEffectPartialDependency_RemoteMlNetGam_FastTreeOnHallOfFameBallot_SB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, variable = "SB")
#GeneralizedAdditiveModels
varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_SB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, variable = "SB")
varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_SB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, variable = "SB")
#LightGbm
varEffectPartialDependency_RemoteMlNetGam_LightGbmInductedToHallOfFame_SB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, variable = "SB")
varEffectPartialDependency_RemoteMlNetGam_LightGbmOnHallOfFameBallot_SB  <- variable_effect_partial_dependency(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, variable = "SB")
#Plots
plot(varEffectPartialDependency_RemoteMlNetGam_FastTreeInductedToHallOfFame_SB,varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsInductedToHallOfFame_SB,varEffectPartialDependency_RemoteMlNetGam_LightGbmInductedToHallOfFame_SB)
plot(varEffectPartialDependency_RemoteMlNetGam_FastTreeOnHallOfFameBallot_SB, varEffectPartialDependency_RemoteMlNetGam_GeneralizedAdditiveModelsOnHallOfFameBallot_SB, varEffectPartialDependency_RemoteMlNetGam_LightGbmOnHallOfFameBallot_SB)

# f) Global Explainer - Model Diagnostics - Model Comparison
?DALEXtra::overall_comparison
overallComparisonOnHallOfFameBallot <- DALEXtra::overall_comparison(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, list(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot), type="classification")
plot(overallComparisonOnHallOfFameBallot)

#?DALEXtra::funnel_measure
funnelLightGbmeOnHallOfFameBallot_vsOther <- DALEXtra::funnel_measure(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, list(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, explainer_RemoteMlNet_FastTreeOnHallOfFameBallot), measure_function = loss_root_mean_square, nbins = 5)
savePlot("GlobalExplainer-ModelDiagnostics-ModelComparison-OnHallOfFameBallot", plot(funnelLightGbmeOnHallOfFameBallot_vsOther)[[1]], TRUE)
funnelLightGbmeInductedToHallOfFame_vsOther <- DALEXtra::funnel_measure(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, list(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, explainer_RemoteMlNet_FastTreeInductedToHallOfFame), measure_function = loss_root_mean_square, nbins = 5)
savePlot("GlobalExplainer-ModelDiagnostics-ModelComparison-InductedToHallOfFame", plot(funnelLightGbmeInductedToHallOfFame_vsOther)[[1]], TRUE)

######################################################
# 4) MACHINE LEARNING - LOCAL INSTANCE LEVEL MODEL EXPLAINER

# a) Local Instance Explainer - Prediction
mikeTroutData = head(MLBBaseballBatters[MLBBaseballBatters$FullPlayerName == "Ichiro Suzuki",], 1)
prediction_RemoteMlNet_FastTreeOnHallOfFameBallot_mikeTrout <- getBaseballHofPrediction(httpObject_FastTreeOnHallOfFameBallot, mikeTroutData)
prediction_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot_mikeTrout <- getBaseballHofPrediction(httpObject_GeneralizedAdditiveModelsOnHallOfFameBallot, mikeTroutData)
prediction_RemoteMlNet_LightGbmOnHallOfFameBallot_mikeTrout <- getBaseballHofPrediction(httpObject_LightGbmOnHallOfFameBallot, mikeTroutData)
print(prediction_RemoteMlNet_FastTreeOnHallOfFameBallot_mikeTrout)
print(prediction_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot_mikeTrout)
print(prediction_RemoteMlNet_LightGbmOnHallOfFameBallot_mikeTrout)
prediction_RemoteMlNet_FastTreeInductedToHallOfFame_mikeTrout <- getBaseballHofPrediction(httpObject_FastTreeInductedToHallOfFame, mikeTroutData)
prediction_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame_mikeTrout <- getBaseballHofPrediction(httpObject_GeneralizedAdditiveModelsInductedToHallOfFame, mikeTroutData)
prediction_RemoteMlNet_LightGbmInductedToHallOfFame_mikeTrout <- getBaseballHofPrediction(httpObject_LightGbmInductedToHallOfFame, mikeTroutData)
print(prediction_RemoteMlNet_FastTreeInductedToHallOfFame_mikeTrout)
print(prediction_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame_mikeTrout)
print(prediction_RemoteMlNet_LightGbmInductedToHallOfFame_mikeTrout)

mikeTroutData = head(MLBBaseballBatters[MLBBaseballBatters$FullPlayerName == "Mike Trout",], 1)
prediction_RemoteMlNet_FastTreeOnHallOfFameBallot_mikeTrout <- getBaseballHofPrediction(httpObject_FastTreeOnHallOfFameBallot, mikeTroutData)
prediction_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot_mikeTrout <- getBaseballHofPrediction(httpObject_GeneralizedAdditiveModelsOnHallOfFameBallot, mikeTroutData)
prediction_RemoteMlNet_LightGbmOnHallOfFameBallot_mikeTrout <- getBaseballHofPrediction(httpObject_LightGbmOnHallOfFameBallot, mikeTroutData)
print(prediction_RemoteMlNet_FastTreeOnHallOfFameBallot_mikeTrout)
print(prediction_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot_mikeTrout)
print(prediction_RemoteMlNet_LightGbmOnHallOfFameBallot_mikeTrout)
prediction_RemoteMlNet_FastTreeInductedToHallOfFame_mikeTrout <- getBaseballHofPrediction(httpObject_FastTreeInductedToHallOfFame, mikeTroutData)
prediction_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame_mikeTrout <- getBaseballHofPrediction(httpObject_GeneralizedAdditiveModelsInductedToHallOfFame, mikeTroutData)
prediction_RemoteMlNet_LightGbmInductedToHallOfFame_mikeTrout <- getBaseballHofPrediction(httpObject_LightGbmInductedToHallOfFame, mikeTroutData)
print(prediction_RemoteMlNet_FastTreeInductedToHallOfFame_mikeTrout)
print(prediction_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame_mikeTrout)
print(prediction_RemoteMlNet_LightGbmInductedToHallOfFame_mikeTrout)

# b) Local Instance Explainer - Prediction Parts | Breakdown
predictionBreakdown_RemoteMlNet_FastTreeOnHallOfFameBallot_MikeTrout <- predict_parts_break_down(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, new_observation = mikeTroutData)
predictionBreakdown_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot_MikeTrout <- predict_parts_break_down(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, new_observation = mikeTroutData)
predictionBreakdown_RemoteMlNet_LightGbmOnHallOfFameBallot_MikeTrout <- predict_parts_break_down(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, new_observation = mikeTroutData)
predictionBreakdown_RemoteMlNet_FastTreeInductedToHallOfFame_MikeTrout <- predict_parts_break_down(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, new_observation = mikeTroutData)
predictionBreakdown_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame_MikeTrout <- predict_parts_break_down(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, new_observation = mikeTroutData)
predictionBreakdown_RemoteMlNet_LightGbmInductedToHallOfFame_MikeTrout <- predict_parts_break_down(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, new_observation = mikeTroutData)
savePlot("LocalExplainer-PredictionParts-BreakDown-OnHallOfFameBallot-FastTree-MikeTrout", plot(predictionBreakdown_RemoteMlNet_FastTreeOnHallOfFameBallot_MikeTrout), TRUE)
savePlot("LocalExplainer-PredictionParts-BreakDown-OnHallOfFameBallot-GeneralizedAdditiveModels-MikeTrout", plot(predictionBreakdown_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot_MikeTrout), TRUE)
savePlot("LocalExplainer-PredictionParts-BreakDown-OnHallOfFameBallot-LightGbm-MikeTrout", plot(predictionBreakdown_RemoteMlNet_LightGbmOnHallOfFameBallot_MikeTrout), TRUE)
savePlot("LocalExplainer-PredictionParts-BreakDown-InductedToHallOfFame-FastTree-MikeTrout", plot(predictionBreakdown_RemoteMlNet_FastTreeInductedToHallOfFame_MikeTrout), TRUE)
savePlot("LocalExplainer-PredictionParts-BreakDown-InductedToHallOfFame-GeneralizedAdditiveModels-MikeTrout", plot(predictionBreakdown_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame_MikeTrout), TRUE)
savePlot("LocalExplainer-PredictionParts-BreakDown-InductedToHallOfFame-LightGbm-MikeTrout", plot(predictionBreakdown_RemoteMlNet_LightGbmInductedToHallOfFame_MikeTrout), TRUE)

# c) Local Instance Explainer - Prediction Parts | Breakdown Interactions
#predictionBreakdownInteractions_RemoteMlNet_FastTreeOnHallOfFameBallot_MikeTrout <- predict_parts_break_down_interactions(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, new_observation = mikeTroutData)
#predictionBreakdownInteractions_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot_MikeTrout <- predict_parts_break_down_interactions(explainer_RemoteMlNet_GeneralizedAdditiveModelsOnHallOfFameBallot, new_observation = mikeTroutData)
#predictionBreakdownInteractions_RemoteMlNet_LightGbmOnHallOfFameBallot_MikeTrout <- predict_parts_break_down_interactions(explainer_RemoteMlNet_LightGbmOnHallOfFameBallot, new_observation = mikeTroutData)
#predictionBreakdownInteractions_RemoteMlNet_FastTreeInductedToHallOfFame_MikeTrout <- predict_parts_break_down_interactions(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, new_observation = mikeTroutData)
#predictionBreakdownInteractions_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame_MikeTrout <- predict_parts_break_down_interactions(explainer_RemoteMlNet_GeneralizedAdditiveModelsInductedToHallOfFame, new_observation = mikeTroutData)
#predictionBreakdownInteractions_RemoteMlNet_LightGbmInductedToHallOfFame_MikeTrout <- predict_parts_break_down_interactions(explainer_RemoteMlNet_LightGbmInductedToHallOfFame, new_observation = mikeTroutData)
#plot(predictionBreakdownInteractions_RemoteMlNet_FastTreeOnHallOfFameBallot_MikeTrout)

# d) Local Instance Explainer - Prediction Profile | Cateris Paribus
?ingredients::ceteris_paribus_2d
caterisParibus_RemoteMlNet_FastTreeOnHallOfFameBallot_MikeTrout_HRvRBI <- ingredients::ceteris_paribus_2d(explainer_RemoteMlNet_FastTreeOnHallOfFameBallot, mikeTroutData, variables = c("HR", "RBI"))
savePlot("LocalExplainer-PredictionProfile-CeterisParibus-OnHallOfFameBallot-FastTree-MikeTrout-HRvRBI", plot(caterisParibus_RemoteMlNet_FastTreeOnHallOfFameBallot_MikeTrout_HRvRBI), TRUE)
caterisParibus_RemoteMlNet_FastTreeInductedToHallOfFame_MikeTrout_HvTotalPlayerAwards <- ingredients::ceteris_paribus_2d(explainer_RemoteMlNet_FastTreeInductedToHallOfFame, mikeTroutData, variables = c("H", "TotalPlayerAwards"))
savePlot("LocalExplainer-PredictionProfile-CeterisParibus-OnHallOfFameBallot-FastTree-MikeTrout-HvTotalPlayerAwards", plot(caterisParibus_RemoteMlNet_FastTreeInductedToHallOfFame_MikeTrout_HvTotalPlayerAwards), TRUE)
