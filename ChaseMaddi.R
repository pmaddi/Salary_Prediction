#Final Project
#Predicting Salries in the UK
#Patrick Chase
#Pranav Maddi

library(class)
library(MASS)
library(nnet)
library(klaR)
library(rpart)
library(randomForest)
library(gbm)
library(rpart.plot)
library(tm)
library(irlba)
library(tree)
library(RGraphics)
library(kknn)

#Read in train data
data = read.csv("Train_rev1.csv")


################### ADD LOCATION VARIABLE ###################

locData = read.csv("locations.csv", header = TRUE, fill = TRUE, row.names = NULL)
locs = rep(0, nrow(data))

for(i in 1:nrow(data))
{
  loc = data$LocationNormalized[i]
  indexLoc = which(as.character(loc) == as.character(locData$location3))
  if(length(indexLoc) != 0)
  {
    locs[i] = locData$city[indexLoc[1]]
  }
  else
  {
    indexLoc = which(as.character(loc) == as.character(locData$location2))
    if(length(indexLoc) != 0)
    {
      locs[i] = locData$city[indexLoc[1]]
    }
    else
    {
      indexLoc = which(as.character(loc) == as.character(locData$location1))
      if(length(indexLoc) != 0)
      {
        locs[i] = locData$city[indexLoc[1]]
      }
      else
      {
        indexLoc = which(as.character(loc) == as.character(locData$region1))
        if(length(indexLoc) != 0)
        {
          locs[i] = locData$city[indexLoc[1]]
        }
        else
        {
          indexLoc = which(as.character(loc) == as.character(locData$city))
          if(length(indexLoc) != 0)
          {
            locs[i] = locData$city[indexLoc[1]]
          }
          else
          {
            locs[i] = locData$country[1]
          }
        }
      }
    }
  }
}

################ END ADD LOCATION VARIABLE ##################

############### HISTOGRAMS ####################

histogram(data$SalaryNormalized)

histogram(log(data$SalaryNormalized), main = "Histogram of Log(Salaries)")

############### END HISTOGRAMS #################

################# BOX PLOTS ##########################

#boxplot for ContractType
boxplot(log(data$SalaryNormalized) ~ data$ContractType, xlab = "ContractType", ylab = "log(Salary)", main = "Box Plots for ContractType")
levels(data$ContractType) = c("not_listed", "full_time", "part_time")

#boxplot for ContractTime
boxplot(log(data$SalaryNormalized) ~ data$ContractTime, xlab = "ContractTime", ylab = "log(Salary)",main = "Box Plots for ContractTime")
levels(data$ContractTime) = c("not_listed", "contract", "permanent")


#boxplot for Category
boxplot(log(data$SalaryNormalized) ~ data$Category, xlab = "Category", ylab = "log(Salary)", main= "Box Plots for Category")
levelsCat = levels(data$Category)
levels(data$Category) = seq(1:29)
levels(data$Category) = levelsCat


# compute index of ordered 'cost factor' and reassign   
DF = data
oind <- order(as.numeric(by(DF$SalaryNormalized, DF$Category, median)))    
DF$Category <- ordered(DF$Category, levels=levels(DF$Category)[oind])   
boxplot(log(SalaryNormalized) ~ Category, data=DF, xlab = "Category", ylab = "log(Salary)", main= "Sorted Box Plots for Category")
levelsCatS = levels(DF$Category)
levels(DF$Category) = seq(1:29)
levels(DF$Category) = levelsCatS

#boxplot for Location
boxplot(log(data$SalaryNormalized) ~ data$locs, xlab = "Location", ylab = "log(Salary)")

################### END BOX PLOTS ####################


############## SPLIT DATA ############################

sizeTrainSample = 20000
sizeTestSample = 10000

#take random sample of training data
sampleIndices = sample(length(data[,1]), sizeTrainSample, replace = FALSE)
trainData = data[sampleIndices,]
train = trainData

#take random sample of testing data
sampleIndices = sample(length(data[,1]), sizeTestSample, replace = FALSE)
testData = data[sampleIndices,]
test = testData

################# END SPLITING DATA ########################



############# BUILDING DTM ##################################

# build a corpus of FullDescriptions
train.corpus <- Corpus(VectorSource(train$FullDescription))

# make each letter lowercase
train.corpus <- tm_map(train.corpus, tolower) 

# remove punctuation 
train.corpus <- tm_map(train.corpus, removePunctuation)

# remove generic and custom stopwords
my_stopwords <- c(stopwords('english'))
train.corpus <- tm_map(train.corpus, removeWords, my_stopwords)



# build a term-document matrix
train.dtm.full <- TermDocumentMatrix(train.corpus)

#use TfIdf weighting instead of just frequency
train.dtm.full = weightBin(train.dtm.full)

inspect(train.dtm.full)

################# END BUILD DTM ##############################

################### REMOVE SPARSE TERMS ######################

# remove sparse terms to simplify the cluster plot
# Note: tweak the sparse parameter to determine the number of words
# sparse is maximum sparsity allowed
# About 10-30 words is good.
train.dtm.shrunk <- removeSparseTerms(train.dtm.full, sparse = 0.85)

inspect(train.dtm.shrunk)

train.df = as.data.frame(t(as.data.frame(inspect(train.dtm.shrunk))))

# inspect dimensions of the data frame
nrow(train.df)
ncol(train.df)

################### END REMOVE SPARSE TERMS ###############

############## BUILD DTM FOR TEST DATA ####################

# build a corpus of FullDescriptions
test.corpus <- Corpus(VectorSource(testData$FullDescription))

test.dtm = TermDocumentMatrix(test.corpus, control = list(dictionary = colnames(train.df)))
test.dtm = weightBin(test.dtm)

inspect(test.dtm)

test.df = as.data.frame(t(as.data.frame(inspect(test.dtm))))

############ END BUILD DTM FOR TEST DATA ##################

############### ADD BACK OTHER VARIABLES #################

train = train.df

train$ContractType = trainData$ContractType
train$ContractTime = trainData$ContractTime
#Company = train$Company
train$Category = trainData$Category
train$Location = trainData$locs
train$Salaries = trainData$SalaryNormalized

nrow(train)
ncol(train)

test = test.df

test$ContractType = testData$ContractType
test$ContractTime = testData$ContractTime
#Company = test$Company
test$Location = testData$locs
test$Category = testData$Category
test$Salaries = testData$SalaryNormalized

nrow(test)
ncol(test)

########## END ADD BACK OTHER VARIABLES ###############

############ RUN MEAN BENCHMARK #####################
# Predict on Training Data
yhat = rep(mean(train$Salaries),nrow(train))
y <- trainData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
# [1] 13397.79
# Predict on Testing Data
yhat = rep(mean(train$Salaries),nrow(test))
y <- testData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
# [1] 13802.76
############ END MEAN BENCHMARK #####################

########## RUN TREE REGRESSION #######################

#tree growing
train.tr <- rpart(Salaries ~., data = train, cp = 0.0017)
rpart.plot(train.tr)
plotcp(train.tr)
train.tr

#tree pruning
train.tr.pr = prune(train.tr, cp = .004)
#windows()
plot(train.tr.pr)
rpart.plot(train.tr.pr, cex = 0.5)
text(train.tr.pr)
plotcp(train.tr.pr)


yhat1 = predict(train.tr.pr, train)
y <- trainData$SalaryNormalized
sum(sqrt((yhat1 - y)^2)/length(y))
#[1] 10621.68

yhat2 = predict(train.tr.pr, test)
y <- testData$SalaryNormalized
sum(sqrt((yhat2 - y)^2)/length(y))
#[1] 10900.48


########## END TREE REGRESSION #######################

############ RUN RANDOM FOREST REGRESSION #####################

#Forloop, vary ntrees, and ncandidate input vars
ntrys = c(2,5,sqrt(95),12,25,95)
tuneMat = matrix(0,150,length(ntrys))
colnames(tuneMat) = ntrys
for (i in seq(1,length(ntrys))) 
{
  rf = randomForest(Salaries ~ ., train, ntree = 150, mtry = ntrys[i])
  #tuneMat[,i] = rf$mse
  #print(tuneMat)
}

yhat <- predict(rf,test)
matplot(1:150, tuneMat ,xlab = "Number of Trees", ylab = "CV Training Error", type="l",main="Random Forest Tuning Parameters")
legend(x="topright", legend = ntrys, col=1:6, lty=1:6, merge=T, trace=F, title = "ntrys")


# Generate Random Forest model
rf = randomForest(Salaries ~ ., train, ntree = 150, mtry = 5)

# Predict on Training Data
yhat = predict(rf,train)
y <- trainData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
#[1] 4165.228

# Predict on Testing Data
yhat = predict(rf,test)
y <- testData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
#[1] 9564.45

# Importance Plot
plot(importance(rf))
sort(importance(rf),index.return = T)

barplot(rev(t(importance(rf))),horiz=F,col="green")
barplot(sort(t(importance(rf))),horiz=F,col="green",main="Random Forest Importance")

########## END RANDOM FOREST REGRESSION ###############


########## LOG TRANSFORM SALARIES FOR LINEAR METHODS ###############
train$Salaries = log(train$Salaries)
test$Salaries = log(test$Salaries)
########## END LOG TRANSFORM SALARIES FOR LINEAR METHODS ##########


############ RUN SIMPLE LINEAR REGRESSION #####################
g <- lm(Salaries ~ ., train)
# Predict on Training Data
yhat = exp(predict(g,train))
y <- trainData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
#[1] 9420.009
# Predict on Testing Data
yhat = exp(predict(g,test))
y <- testData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
#[1] 9937.49
########### END SIMPLE LINEAR REGRESSION #####################


############ VARIABLE SELECTION METHODS #####################

############ RUN LASSO #####################
library(lars)
# Run the cross validation and find the best lambda value
g.lasso.cv <- cv.lars(x=as.matrix(train[,-ncol(train)]),y=as.matrix(train[,ncol(train)]),type="lasso")
lam = g.lasso.cv$index[g.lasso.cv$cv==min(g.lasso.cv$cv)]
# Generate lasso model
tr.lasso <- lars(x=as.matrix(train[,-ncol(train)]),y=as.matrix(train[,ncol(train)]),type="lasso")
# Predict on training data
train.lasso <- exp(predict.lars(tr.lasso, train[,-ncol(train)], s=lam, type="fit",mode="fraction"))
y <- trainData$SalaryNormalized
sqrt(sum((train.lasso$fit - y)^2)/length(y))
# Predict on test data
test.lasso <- exp(predict.lars(tr.lasso, test[,-ncol(test)], s=lam, type="fit",mode="fraction"))
y <- testData$SalaryNormalized
sqrt(sum((test.lasso$fit - y)^2)/length(y))
coef <- predict.lars(tr.lasso, test[,-ncol(communities)], s=lam, type="coef",mode="fraction")$coef
# Number of used coefficents
sum(coef!=0)
############ END LASSO #####################

############ RUN RIDGE REGRESSION #####################
# Cross Validate
cv.ridge <- function(formula,data, lam, Kfold=10, seed=seed)
{
  n= nrow(data); p=ncol(data)
  set.seed(seed)
  id <- sample(1:n);data1=data[id,]
  group <- rep(1:Kfold, n/Kfold+1)[1:n]
  yhat = matrix(0, nrow(data), length(lam))
  for (i in 1:Kfold)
  {
    test <- data1[group==i,]
    train <- data1[group!=i,]
    result <- lm.ridge(formula, train,lam=lam)
    test.x <- scale(test[,-p])
    coef <- result$coef
    yhat[group==i,] <- mean(train[,p])+test.x%*%coef
  }
  return(apply(data1[,p]-yhat, 2, function(x)return(mean(x^2))))
}
cvridge = cv.ridge(Salaries~., train, lam=seq(1,4000,100), seed=134)
plot(seq(1,4000,100), sqrt(cvridge),xlab="lambda--full dataset",main="Ridge Regression Lambda Cross Validation")
which.min(cvridge)
# Generate Model
g.rg <- lm.ridge(Salaries~., train, lam=(seq(1,4000,100))[which.min(cvridge)])
# Test Error
test.x <- scale(test[,-ncol(test)])
yhat.test <- exp(mean(train[,ncol(train)])+test.x%*%g.rg$coef)
sqrt(sum((test[,ncol(test)]-yhat.test)^2)/length(yhat.test))
# Training Error
train.x <- scale(train[,-ncol(train)])
yhat.training <- exp(mean(train[,ncol(train)])+train.x%*%g.rg$coef)
sqrt(sum((train[,ncol(train)]-yhat.training)^2)/length(yhat.training))
length(g.rg$coef)
############ END RIDGE REGRESSION #####################

############ END VARIABLE SELECTION METHODS #####################

########## UNTRANSFORM SALARIES ###############
train$Salaries = exp(train$Salaries)
test$Salaries = exp(test$Salaries)
########## END UNTRANSFORM SALARIES ###################

########### RUN K MEANS ###################################
train.kknn = kknn(Salaries ~ ., train, train, k = 10)
# Predict on Training Data
yhat = train.kknn$fitted.values
y <- trainData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
#[1] 5752.845

test.kknn = kknn(Salaries ~ ., train, test, k = 10)
# Predict on Testing Data
yhat = test.kknn$fitted.values
y <- testData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
#[1] 11619.04

########### END RUN K MEANS ###################################

############ PCA ON TRAINING DATA#####################
numberOfPCAVariables = 20

#matrix of entire training information
train.df.full = t(as.data.frame(inspect(train.dtm.full)))

#matrix of entire testing information with same dictionary as training
test.dtm.full = TermDocumentMatrix(test.corpus, control = list(dictionary = colnames(train.df.full)))
test.dtm.full = weightBin(test.dtm.full, normalize = TRUE)

inspect(test.dtm.full)

test.df.full = t(as.data.frame(inspect(test.dtm.full)))

#calculate PCA of training
train.pca = prcomp(train.df.full, retx = TRUE)
redMatrix = train.pca$x[,1:numberOfPCAVariables]

train.PCA = as.data.frame(redMatrix)

#add variables to PCA train columns
train.PCA$Salaries = trainData$SalaryNormalized
train.PCA$ContractType = trainData$ContractType
train.PCA$ContractTime = trainData$ContractTime
#Company = trainData$Company
train.PCA$Location = trainData$loc
train.PCA$Category = trainData$Category

#calculate PCA of test
test.pca = prcomp(test.df.full, retx = TRUE)
redMatrix = test.pca$x[,1:numberOfPCAVariables]

test.PCA = as.data.frame(redMatrix)

#add variables to PCA test columns
test.PCA$Salaries = testData$SalaryNormalized
test.PCA$ContractType = testData$ContractType
test.PCA$ContractTime = testData$ContractTime
#Company = testData$Company
test.PCA$Location = testData$loc
test.PCA$Category = testData$Category


############ END PCA ON TRAINING DATA #################

############# TREE ON PCA #############################

train = train.PCA

test = test.PCA

#tree growing
train.tr <- rpart(Salaries ~., data = train.PCA, cp = 0.003)
rpart.plot(train.tr)
plotcp(train.tr)
train.tr

#tree pruning
train.tr.pr = prune(train.tr, cp = .007)
#windows()
plot(train.tr.pr)
rpart.plot(train.tr.pr)
text(train.tr.pr)
plotcp(train.tr.pr)

yhat1 = predict(train.tr.pr, train)
y <- trainData$SalaryNormalized
sum(sqrt((yhat1 - y)^2)/length(y))
#10274.61

yhat2 = predict(train.tr.pr, test)
y <- testData$SalaryNormalized
sum(sqrt((yhat2 - y)^2)/length(y))
#11288.89


################ END TREE ON PCA ######################

############## RANDOM FOREST PCA ######################

# < Cross Validate for nTry here! > #
# Generate Random Forest model
rf = randomForest(Salaries ~ ., train, ntree = 150, ntry = sqrt(ncol(train)))

# Predict on Training Data
yhat = predict(rf,train)
y <- train$Salaries
sum(sqrt((yhat - y)^2)/length(y))

# Predict on Testing Data
yhat = predict(rf,test)
y <- test$Salaries
sum(sqrt((yhat - y)^2)/length(y))

# Importance Plot
plot(importance(rf))
sort(importance(rf),index.return = T)

############## END RANDOM FOREST PCA ###################

########## LOG TRANSFORM SALARIES FOR LINEAR METHODS ###############
train$Salaries = log(train$Salaries)
test$Salaries = log(test$Salaries)
########## END LOG TRANSFORM SALARIES FOR LINEAR METHODS ##########

############ LINEAR REGRESSION ON PCA #####################
g <- lm(Salaries ~ ., train)
# Predict on Training Data
yhat = exp(predict(g,train))
y <- trainData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
#[1] 9420.009

# Predict on Testing Data
yhat = exp(predict(g,test))
y <- testData$SalaryNormalized
sum(sqrt((yhat - y)^2)/length(y))
#[1] 9937.49
########### END LINEAR REGRESSION ON PCA #####################

########## UNTRANSFORM SALARIES ###############
train$Salaries = exp(train$Salaries)
test$Salaries = exp(test$Salaries)
########## END UNTRANSFORM SALARIES ###################
