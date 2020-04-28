# Christopher Sandoval 13660
# Maria Fernanda Estrada 14198

# Instalacion de paquetes
install.packages("neural")
install.packages("dummy")
install.packages("nnet")
install.packages("RWeka")
install.packages("neuralnet")

# Librerias necesarias
library(caret)
library(nnet)
library(RWeka)
library(neural)
library(dummy)
library(neuralnet)
library(e1071)




# Importando datos de entrenamiento y limpieza
data_training <- read.csv("train.csv", stringsAsFactors = FALSE)
data_training$Class <- as.factor(ifelse(data_training$SalePrice >= 270000, "Cara", ifelse(data_training$SalePrice >= 195000, "Intermedia", "Economica")))
data_training_filtered <- data_training[, c(2,19,20,35,45,48,52,71,82)]

# Importando datos de test y limpieza
data_test <- read.csv("test.csv", stringsAsFactors = FALSE)
data_test_filtered <- data_test[, c(2,19,20,35,45,48,52,71)]
data_sample <- read.csv("sample_submission.csv", stringsAsFactors = FALSE)
data_sample$Class <- as.factor(ifelse(data_sample$SalePrice >= 270000, "Cara", ifelse(data_sample$SalePrice >= 195000, "Intermedia", "Economica")))
data_test_filtered$Class <- data_sample$Class
data_test_filtered <- na.omit(data_test_filtered)



#-------------------------------------------------
# Red Neuronal con caret
#-------------------------------------------------

# Modelo 1 usando nnet
modeloCaret <- train(Class~., data=data_training_filtered, method="nnet", trace=F)
data_test_filtered$prediccionCaret<-predict(modeloCaret, newdata = data_test_filtered[,1:8])
cfmCaret<-confusionMatrix(data_test_filtered$prediccionCaret,data_test_filtered$Class)
cfmCaret

# Modelo 2 usando pcaNNet
modeloCaret2 <- train(Class~., data=data_training_filtered, method="pcaNNet", trace=F)
data_test_filtered$prediccionCaret2<-predict(modeloCaret2, newdata = data_test_filtered[,1:8])
cfmCaret2<-confusionMatrix(data_test_filtered$prediccionCaret2,data_test_filtered$Class)
cfmCaret2



#-------------------------------------------------
# SVM
#-------------------------------------------------

# Modelo 1: lineal y c=2^5
modeloSVM_1<-svm(Class~., data=data_training_filtered, cost=2^5, kernel="linear")
prediccion_1<-predict(modeloSVM_1,newdata=data_training_filtered[,1:8])
confusionMatrix(data_training_filtered$Class,prediccion_1)

# Modelo 2: lineal y c=2^-5
modeloSVM_2<-svm(Class~., data=data_training_filtered, cost=2^-5, kernel="linear")
prediccion_2<-predict(modeloSVM_2,newdata=data_training_filtered[,1:8])
confusionMatrix(data_training_filtered$Class,prediccion_2)

# Modelo 3: radial y gamma=2^-5
modeloSVM_3<-svm(Class~., data=data_training_filtered, gamma=2^-5, kernel="radial")
prediccion_3<-predict(modeloSVM_3,newdata=data_training_filtered[,1:8])
confusionMatrix(data_training_filtered$Class,prediccion_3)
