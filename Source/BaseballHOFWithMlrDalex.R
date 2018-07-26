# TESTED ON MRO 3.5.0 & CRAN R 3.5.1

# 1) INSTALL & LOAD PACKAGES

# a) INSTALL Core Modeling Packages
if('kernlab' %in% rownames(installed.packages()) == FALSE) {install.packages('kernlab')}
if('randomForest' %in% rownames(installed.packages()) == FALSE) {install.packages('randomForest')}
if('xgboost' %in% rownames(installed.packages()) == FALSE) {install.packages('xgboost')}
# ML Orchestration/Helper Packages
if('DALEX' %in% rownames(installed.packages()) == FALSE) {install.packages('DALEX')}
if('mlr' %in% rownames(installed.packages()) == FALSE) {install.packages('mlr')}

# b) LOAD Package
# Core Modeling Packages
library(kernlab)
library(randomForest)
library(xgboost)
# Core Modeling Packages
library(DALEX)
library(mlr)


# 2) SETUP TRAINING, VALIDATION & COMBINED DATA SETS

# a) Set working directory
getwd()
wdPath <- "C:\\Users\\bart\\Downloads\\BaseballHOFPredictionWithMlrAndDALEX-master\\BaseballHOFPredictionWithMlrAndDALEX-master\\Source"
setwd(wdPath)

# b) Load Training Data CSV Files
trainingData <- read.csv(file="HOFTrainingWithHeader.csv", header=TRUE, sep=",")
validationData <- read.csv(file="HOFValidationWithHeader.csv", header=TRUE, sep=",")
combinedData <- rbind(trainingData, validationData)

# c) Change InductedToHallOfFame as a 1/0 binary factor
trainingData$InductedToHallOfFame <- factor(ifelse(trainingData$InductedToHallOfFame =='TRUE', 1, 0))
validationData$InductedToHallOfFame <- factor(ifelse(validationData$InductedToHallOfFame =='TRUE', 1, 0))
combinedData$InductedToHallOfFame <- factor(ifelse(combinedData$InductedToHallOfFame =='TRUE', 1, 0))

# d) Copy data set to make predictions in bottom of script, before removing player Name and ID
# This is so we can easily lookup players by ID or FullPlayerName later
fullPlayerData <- combinedData

# e) Remove ID columns (FullPLayerName, LastYearPlayed, ID) not needed for training classifier models
trainingData <- subset(trainingData, select = -c(FullPlayerName, LastYearPlayed, ID))
validationData <- subset(validationData, select = -c(FullPlayerName, LastYearPlayed, ID))
combinedData <- subset(combinedData, select = -c(FullPlayerName, LastYearPlayed, ID))
nrow(trainingData)
nrow(validationData)
nrow(combinedData)

# f) Change InductedToHallOfFame to a factor for model training
# trainingData$InductedToHallOfFame <- factor(trainingData$InductedToHallOfFame)
# validationData$InductedToHallOfFame <- factor(validationData$InductedToHallOfFame)

# g) Check Validation Data
head(validationData, 15)

# 3) MACHINE LEARNING - TRAINING

# a) Set the seed explicitly for training reproducability
set.seed(12345)

# b) Create (binary) classification tasks (MLR package construct)
classif_task <- makeClassifTask(id = "class1", data = trainingData, target = "InductedToHallOfFame", positive = 1)
classif_task_combined <- makeClassifTask(id = "class2", data = combinedData, target = "InductedToHallOfFame", positive = 1)

# C) Make Learners - Random Forest, GLM, XgBoost (MLR package construct)
classif_lrn_rf <- makeLearner("classif.randomForest", predict.type = "prob")
classif_lrn_glm <- makeLearner("classif.binomial", predict.type = "prob")
classif_lrn_xgboost <- makeLearner("classif.xgboost", predict.type = "prob")
xgboostParams <- list(nrounds = 400, max_depth = 30, nthread = 4, max_delta_step = 6,
                      num_parallel_tree = 4, eta=0.02, gamma = 1)
classif_lrn_xgboost <- makeLearner("classif.xgboost", predict.type = "prob", par.vals = xgboostParams)

# d) Traing the classification models using learners (and their accompanying parameters)
?configureMlr
getMlrOptions()
classif_rf <- train(classif_lrn_rf, classif_task)
classif_glm <- train(classif_lrn_glm, classif_task)
classif_xgboost <- train(classif_lrn_xgboost, classif_task)
classif_xgboostTwo <- train(classif_lrn_xgboost, classif_task_combined)

# e) Make predictions on validation data
predRf = predict(classif_rf, newdata = validationData)
predGlm = predict(classif_glm, newdata = validationData)
predXgBoost = predict(classif_xgboost, newdata = validationData)
predXgBoostTwo = predict(classif_xgboostTwo, newdata = validationData)

# 4) MACHINE LEARNING - EVALUATION

# a) Test Performance of classifiers (models) using various performance metrics
# You can get a list of available measures
listMeasures("classif")
getDefaultMeasure(classif_lrn_rf)
performanceMetrics <- list(tnr, tpr, lsr, f1, mmce, tp, acc, fdr, kappa, ppv)
performance(predRf, measures = performanceMetrics)
performance(predGlm, measures = performanceMetrics)
performance(predXgBoost, measures = performanceMetrics)

# b) Build a list of important measures to analyze visualy, generate performance data, plot performance data
relevantPerformanceMetrics <- list(tpr, ppv, tp, mmce)

threshholdvsPerf_Rf <- generateThreshVsPerfData(predRf, measures = relevantPerformanceMetrics)
threshholdvsPerf_glm <- generateThreshVsPerfData(predGlm, measures = relevantPerformanceMetrics)
threshholdvsPerf_XgBoost <- generateThreshVsPerfData(predXgBoost, measures = relevantPerformanceMetrics)

# Plot performance metrics
plotThreshVsPerf(threshholdvsPerf_Rf)
plotThreshVsPerf(threshholdvsPerf_glm)
plotThreshVsPerf(threshholdvsPerf_XgBoost)

# c) Calculcate & Show ROC Measures / Confusion Matrix
rocMeasures_Rf <- calculateROCMeasures(predRf)
rocMeasures_Rf
rocMeasures_glm <- calculateROCMeasures(predGlm)
rocMeasures_glm
rocMeasures_XgBoost <- calculateROCMeasures(predXgBoost)
rocMeasures_XgBoost

# Plot ROC Curves
rocCurvesData <- generateThreshVsPerfData(list(rf = predRf, glm = predGlm, xgboost = predXgBoost), 
                              measures = list(fpr, tpr)) 
plotROCCurves(rocCurvesData)

# d) Plot Learner Prediction - All-Star Appearances vs 

# AllStarAppearances vs TB (Total Bases)
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("AllStarAppearances", "TB"), cv = 10)
plotLearnerPrediction(classif_lrn_glm, task = classif_task, features = c("AllStarAppearances", "TB"), cv = 10)
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("AllStarAppearances", "TB"), cv = 10)

# H vs HR
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("H", "HR"), cv = 10)
plotLearnerPrediction(classif_lrn_glm, task = classif_task, features = c("H", "HR"), cv = 10)
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("H", "HR"), cv = 10)

# H vs GoldGloves
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("H", "GoldGloves"), cv = 10)
plotLearnerPrediction(classif_lrn_glm, task = classif_task, features = c("H", "GoldGloves"), cv = 10)
plotLearnerPrediction(classif_lrn_xgboost, task = classif_task, features = c("H", "GoldGloves"), cv = 10)

# MVP vs H
plotLearnerPrediction(classif_lrn_rf, task = classif_task, features = c("MVPs", "H"), cv = 10)



# Hyper Parameters
getParamSet("classif.xgboost")
getParamSet("classif.randomForest")
discrete_ps <- makeParamSet(
  makeDiscreteParam("ntree", values = c(1, 50, 100, 150, 400, 600)),
  makeLogicalParam("replace")
)
tuneControl <- makeTuneControlRandom(maxit = 10L)
rdesc = makeResampleDesc("CV", iters = 3L)
resampling <- rdesc

resultsTuning <- tuneParams("classif.randomForest", task = classif_task, resampling = rdesc,
                            par.set = discrete_ps, 
                            measures = list(ppv, setAggregation(ppv, test.sd)),
                            control = tuneControl)

# Custom Holdout
rin = makeFixedHoldoutInstance(train.inds = 1:6258, test.inds = 6259:9385, size=9385)
rin
resultsTuning <- tuneParams(#"classif.randomForest"
                            classif_lrn_rf, task = classif_task_combined, resampling = rdesc,
                            par.set = discrete_ps, 
                            measures = list(auc, setAggregation(auc, test.sd)),
                            control = tuneControl)

#4) MACHINE LEARNING - MODEL EXPLAINERS
# Uses the DALEX r package

# a) Build a helper function to extract prediction data, required for DALEX explainers
y_test <- as.numeric(as.character(validationData$InductedToHallOfFame))
custom_predict_classif <- function(object, newdata) {pred <- predict(object, newdata=newdata)
{
  response <- pred$data[,3]
  return(response)}  
}

# b) Build model exmplainers for each of the built models
explainer_classif_rf <- DALEX::explain(classif_rf, data=validationData, y=y_test, label= "rf", predict_function = custom_predict_classif)
explainer_classif_glm <- DALEX::explain(classif_glm, data=validationData, y=y_test, label="glm", predict_function = custom_predict_classif)
explainer_classif_xgboost <- DALEX::explain(classif_xgboost, data=validationData, y=y_test, label ="xgboost", predict_function = custom_predict_classif)

# c) Plot Model Performance
?model_performance
mp_classif_rf <- model_performance(explainer_classif_rf)
mp_classif_glm <- model_performance(explainer_classif_glm)
mp_classif_xgboost <- model_performance(explainer_classif_xgboost)
plot(mp_classif_rf, mp_classif_glm, mp_classif_xgboost)

# d) Plot Variable Importance - Individually & Collectively
vi_classif_rf <- variable_importance(explainer_classif_rf, loss_function = loss_root_mean_square)
vi_classif_glm <- variable_importance(explainer_classif_glm, loss_function = loss_root_mean_square)
vi_classif_xgboost <- variable_importance(explainer_classif_xgboost, loss_function = loss_root_mean_square)
plot(vi_classif_rf, vi_classif_glm, vi_classif_xgboost)
plot(vi_classif_xgboost)

# Variable Response - All Star Appearances Feature
vr_AllStarAppearances_rf  <- variable_response(explainer_classif_rf, variable = "AllStarAppearances", type = "pdp")
vr_AllStarAppearances_glm  <- variable_response(explainer_classif_glm, variable = "AllStarAppearances", type = "pdp")
vr_AllStarAppearances_svm  <- variable_response(explainer_classif_svm, variable = "AllStarAppearances", type = "pdp")
vr_AllStarAppearances_xgboost  <- variable_response(explainer_classif_xgboost, variable = "AllStarAppearances", type = "pdp")
plot(vr_AllStarAppearances_rf, vr_AllStarAppearances_glm, vr_AllStarAppearances_svm, vr_AllStarAppearances_xgboost)

# Variable Response - Random Forest Model - HR Feature
vr_hr_rf  <- variable_response(explainer_classif_rf, variable = "HR", type = "pdp")
vr_hr_glm  <- variable_response(explainer_classif_glm, variable = "HR", type = "pdp")
vr_hr_xgboost  <- variable_response(explainer_classif_xgboost, variable = "HR", type = "pdp")
plot(vr_hr_rf, vr_hr_glm, vr_hr_xgboost)
# Variable Response - Random Forest Model - MVP Feature
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "MVPs", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "MVPs", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "MVPs", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)
# Variable Response - Random Forest Model - TB (Total Bases) Feature
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "TB", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "TB", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "TB", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)
# Variable Response - Random Forest Model - GoldGloves Feature
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "GoldGloves", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "GoldGloves", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "GoldGloves", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)


# Prediction Breakdown - Ichiro Suzuki
newPred31 <- validationData[31,]
newPredDownRf31 <- prediction_breakdown(explainer_classif_rf, observation = newPred31)
newPredDownRf31
plot(newPredDownRf31)

# Prediction Breakdown - Kirby Puckett
newPred31 <- validationData[22,]
newPredDownRf31 <- prediction_breakdown(explainer_classif_rf, observation = newPred31)
newPredDownRf31
plot(newPredDownRf31)


# Prediction Breakdown - Willie Mays
newPred12 <- validationData[12,]
newPredDownRf12 <- prediction_breakdown(explainer_classif_rf, observation = newPred12)
newPredDownRf12
plot(newPredDownRf12)

# Prediction Breakdown - Derek Jeter
newPred29 <- validationData[29,]
newPredDownRf29 <- prediction_breakdown(explainer_classif_rf, observation = newPred29)
newPredDownRf29
plot(newPredDownRf29)

# Prediction Breakdown - Tony Perez
newPred18 <- validationData[18,]
newPredDownRf18 <- prediction_breakdown(explainer_classif_rf, observation = newPred18)
newPredDownRf18
plot(newPredDownRf18)


# Prediction Breakdown - Kirby Puckett
kirbyPuckettData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Kirby Puckett",], 1)
prediction_breakdown_KirbyPuckett = prediction_breakdown(explainer_classif_rf, observation = kirbyPuckettData)
plot(prediction_breakdown_KirbyPuckett)

# Prediction Breakdown - Willie Mays
willieMaysData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Willie Mays",], 1)
prediction_breakdown_WillieMays = prediction_breakdown(explainer_classif_rf, observation = willieMaysData)
plot(prediction_breakdown_WillieMays)

# Prediction Breakdown - Derek Jeter
derrekJeterData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Derek Jeter",], 1)
prediction_breakdown_DerekJeter = prediction_breakdown(explainer_classif_rf, observation = derrekJeterData)
plot(prediction_breakdown_DerekJeter)

# Prediction Breakdown - Chase Utley
chaseUtleyData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Chase Utley",], 1)
prediction_breakdown_ChaseUtley = prediction_breakdown(explainer_classif_rf, observation = chaseUtleyData)
plot(prediction_breakdown_ChaseUtley)

# Prediction Breakdown - Jimmy Rollins
jimmyRollinsData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Jimmy Rollins",], 1)
prediction_breakdown_JimmyRollins = prediction_breakdown(explainer_classif_rf, observation = jimmyRollinsData)
plot(prediction_breakdown_JimmyRollins)

# Prediction Breakdown - Jeff Kent
jeffKentData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Jeff Kent",], 1)
prediction_breakdown_JeffKent = prediction_breakdown(explainer_classif_rf, observation = jeffKentData)
plot(prediction_breakdown_JeffKent)

# Prediction Breakdown - Rod Carew
rodCarewData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Rod Carew",], 1)
prediction_breakdown_RodCarew = prediction_breakdown(explainer_classif_rf, observation = rodCarewData)
plot(prediction_breakdown_RodCarew)

# Prediction Breakdown - Roberto Alomar
robertoAlomarData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Roberto Alomar",], 1)
prediction_breakdown_RobertoAlomar = prediction_breakdown(explainer_classif_rf, observation = robertoAlomarData)
plot(prediction_breakdown_RobertoAlomar)

# Prediction Breakdown - Roberto Alomar
ryneSandbergData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Ryne Sandberg",], 1)
prediction_breakdown_RyneSandberg = prediction_breakdown(explainer_classif_rf, observation = ryneSandbergData)
plot(prediction_breakdown_RyneSandberg)

# Prediction Breakdown - Roberto Alomar
robinsonCanoData = head(fullPlayerData[fullPlayerData$FullPlayerName == "Robinson Cano",], 1)
prediction_breakdown_RobinsonCano = prediction_breakdown(explainer_classif_rf, observation = robinsonCanoData)
plot(prediction_breakdown_RobinsonCano)
