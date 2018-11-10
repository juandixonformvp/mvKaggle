library(randomForest)
library(mlbench)
library(caret)
library(e1071)
library(MASS)
library(ggparallel)
library(rpart)
library(devtools)
library(ggbiplot)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(factoextra)
library(MatchIt)
library(splitstackshape)
library(gbm)
library(xgboost)
library(glmnet)

setwd('home/rpchang/mvKaggle')
df <- read.csv('train.csv')
td <- read.csv('test.csv')

df.all <- df
# df.all <- df.all[, -which(colMeans(is.na(df.all)) > 0.000001)] #removes columns with NA values

#remove near zero-variance predictors
remove_cols <- nearZeroVar(df.all, names = TRUE, freqCut = 2, uniqueCut = 20)
all_cols <- names(df.all)
df.all <- df.all[ , setdiff(all_cols, remove_cols)]


df.all$zipcode <- as.factor(df.all$zipcode) #zipcode is integer, but is categorical
df.all <- subset(df.all, select = -c(id, date, lat, long)) #don't need "month sold" and other ID variables
x_pred <- df.all[,2:ncol(df.all)] #drop the y variable
x_pred <- x_pred[ , order(names(x_pred))]
x_train <- model.matrix( ~ .-1, x_pred)


td.all <- td[,names(x_pred)]
td.all$MS.SubClass <- as.factor(td.all$MS.SubClass)
options(na.action="na.pass") # the model matrix function deletes NA observations, must use "na.pass"
y_train <- model.matrix( ~ .-1, td.all)
y_train[is.na(y_train)] <- 0 # now replace NAs with 0

# problem because some categorical levels in training set are not in test set
# convert matrix to dataframe, match columns, then convert back to matrix
x_train <- as.data.frame(x_train)
y_train <- as.data.frame(y_train)

x_train <- x_train[,names(y_train)]

# x_train <- as.matrix(x_train)
# y_train <- as.matrix(y_train)
##########

## code to get interaction terms
f <- as.formula(y ~ .*.)
y <- df.all$SalePrice
x_interact <- model.matrix(f, x_train)[, -1]

f <- as.formula(y ~ .*.)
y <- td$PID #dummy variable, just placeholder
y_interact <- model.matrix(f, y_train)[, -1]

x_interact <- as.matrix(x_interact)
y_interact <- as.matrix(y_interact)

###


fit <-glmnet(x = x_interact, y = df.all$SalePrice, alpha = 1) 
plot(fit, xvar = "lambda")

crossval <-  cv.glmnet(x = x_interact, y = df.all$SalePrice)
plot(crossval)
penalty <- crossval$lambda.min #optimal lambda
penalty #minimal shrinkage
fit1 <-glmnet(x = x_interact, y = df.all$SalePrice, alpha = 1, lambda = penalty ) #estimate the model with that
coef(fit1)

results <- predict(object=fit1, y_interact)
pred <- cbind(td$PID, as.data.frame(results))
colnames(pred)<- c("PID","SalePrice")

write.csv(pred, file = "glmnet_20180127.csv", row.names=FALSE)
