if('DALEX' %in% rownames(installed.packages()) == FALSE) {install.packages('DALEX')}
if('caret' %in% rownames(installed.packages()) == FALSE) {install.packages('caret')}
if('dplyr' %in% rownames(installed.packages()) == FALSE) {install.packages('dplyr')}
if('randomForest' %in% rownames(installed.packages()) == FALSE) {install.packages('randomForest')}

install.packages('DALEX')
install.packages('caret')
install.packages('dplyr')
install.packages('randomForest')

# Load the libraries
library(DALEX)
library(caret)
library(dplyr)
library(randomForest)


wdPath = "/Users/bartczernicki-msft/Desktop/SourceCode/BaseballHOFDalex"
setwd(wdPath)

trainingData <- read.csv(file="HOFTrainingWithHeader.csv", header=TRUE, sep=",")
validationData <- read.csv(file="HOFValidationWithHeader.csv", header=TRUE, sep=",")
trainingData$InductedToHallOfFame <- factor(ifelse(trainingData$InductedToHallOfFame =='TRUE', 1, 0))
validationData$InductedToHallOfFame <- factor(ifelse(validationData$InductedToHallOfFame =='TRUE', 1, 0))
trainingData <- subset(trainingData, select = -c(FullPlayerName, LastYearPlayed, ID))

# summary(trainingData$InductedToHallOfFame)
# summary(validationData$InductedToHallOfFame)
head(trainingData)
head(validationData, 100)

classif_rf <- train(InductedToHallOfFame~., data = trainingData, method="rf", ntree = 100, tuneLength = 1)
classif_glm <- train(InductedToHallOfFame~., data = trainingData, method="glm", family="binomial")
classif_svm <- train(InductedToHallOfFame~., data = trainingData, method="svmRadial", prob.model = TRUE, tuneLength = 1)


p_fun <- function(object, newdata){predict(object, newdata=newdata, type="prob")[,2]}
yTest <- as.numeric(as.character(validationData$InductedToHallOfFame))

explainer_classif_rf <- DALEX::explain(classif_rf, label = "rf",
                                       data = validationData, y = yTest,
                                       predict_function = p_fun)

explainer_classif_glm <- DALEX::explain(classif_glm, label = "glm", 
                                        data = validationData, y = yTest,
                                        predict_function = p_fun)

explainer_classif_svm <- DALEX::explain(classif_svm,  label = "svm", 
                                        data = validationData, y = yTest,
                                        predict_function = p_fun)


mp_classif_rf <- model_performance(explainer_classif_rf)
mp_classif_glm <- model_performance(explainer_classif_glm)
mp_classif_svm <- model_performance(explainer_classif_svm)
plot(mp_classif_rf, mp_classif_glm, mp_classif_svm)
plot(mp_classif_rf, mp_classif_glm, mp_classif_svm, geom = "boxplot")

vi_classif_rf <- variable_importance(explainer_classif_rf, loss_function = loss_root_mean_square)
vi_classif_glm <- variable_importance(explainer_classif_glm, loss_function = loss_root_mean_square)
vi_classif_svm <- variable_importance(explainer_classif_svm, loss_function = loss_root_mean_square)
plot(vi_classif_rf, vi_classif_glm, vi_classif_svm)


pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "HR", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "HR", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "HR", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)

ale_classif_rf  <- variable_response(explainer_classif_rf, variable = "HR", type = "ale")
ale_classif_glm  <- variable_response(explainer_classif_glm, variable = "HR", type = "ale")
ale_classif_svm  <- variable_response(explainer_classif_svm, variable = "HR", type = "ale")
plot(ale_classif_rf, ale_classif_glm, ale_classif_svm)



pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "HR", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "HR", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "HR", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)

pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "MVPs", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "MVPs", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "MVPs", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)

pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "AllStarAppearances", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "AllStarAppearances", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "AllStarAppearances", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)

pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "TB", type = "pdp")
pdp_classif_glm  <- variable_response(explainer_classif_glm, variable = "TB", type = "pdp")
pdp_classif_svm  <- variable_response(explainer_classif_svm, variable = "TB", type = "pdp")
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)

newPred1 <- head(validationData, 1)
newPredDownRf <- prediction_breakdown(explainer_classif_rf, observation = newPred1)
newPredDownRf
plot(newPredDownRf)

newPred2 <- validationData[22,]
newPredDownRf100 <- prediction_breakdown(explainer_classif_rf, observation = newPred2)
newPredDownRf100
plot(newPredDownRf100)

newPred14 <- validationData[14,]
newPredDownRf14 <- prediction_breakdown(explainer_classif_rf, observation = newPred14)
newPredDownRf14
plot(newPredDownRf14)

newPred29 <- validationData[29,]
newPredDownRf29 <- prediction_breakdown(explainer_classif_rf, observation = newPred29)
newPredDownRf29
plot(newPredDownRf29)

newPred31 <- validationData[31,]
newPredDownRf31 <- prediction_breakdown(explainer_classif_rf, observation = newPred31)
newPredDownRf31
plot(newPredDownRf31)



