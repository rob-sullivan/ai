#Multiple Linear Regression

library(caTools)

#importing the data
dataset = read.csv('50_Startups.csv')
# dataset = dataset[, 2:3]

# data preprocessing
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1,2,3))
#split dataset into training and test set
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8) #40 training # 10% in test
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
# not needed in linear reg library

# Fitting Multiple Linear Regression to the Training sete
regressor = lm(formula = Profit ~ ., 
               data = training_set)

# Predicting the Test set Results
y_pred = predict(regressor, newdata = test_set)


#Backward elimination
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
#               data = dataset)
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
#               data = dataset)
#regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
#               data = dataset)
regressor = lm(formula = Profit ~ R.D.Spend, 
               data = dataset)
summary(regressor)

#Automatic Backward Elimination
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)