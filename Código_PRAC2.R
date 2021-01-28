# Limpieza y análisis del Dataset Titanic

##########################################
# 0 Librerias ############################
##########################################
library(readr)
library(data.table)
library(tidyverse)
library(nortest)
library(PerformanceAnalytics)
library(glmnet)
library("nortest")
library(caret)
library(vcd)

##########################################
# 1 Descripción del dataset ##############
##########################################

# Carhgamos solo el dataset de train
titanic <- read.csv("Dataset_original/train.csv")

# vemos la cabecera
head(titanic)

# un resumen de los datos
str(titanic)

##########################################
## 3.1 Valores perdidos###################
##########################################

# Unificamos por si acaso todos los posibles valores perdidos
unify_null <- function(x){
  na_index<-(is.na(x) | x=="null" | x=="NULL" | x=='\\N' | x=='\\n' | x=="" | x=="?" | x=='NA');
  x[na_index]<-NA;
  return(x)
}
#aplicamos la funcion
titanic <- data.table(apply(titanic,2,unify_null))

# Vemos cuantos valores perdido tiene cada variable
sapply(titanic, function(x){sum(is.na(x))})
# Porcentaje de valores perdidos
sapply(titanic, function(x){100*sum(is.na(x))/length(x)})

#Eliminamos la variable Cabin ya que tiene muchos perdidos y no nos va a ser util
titanic <- titanic[,-11]

# Reemplazamos los valores perdidos
# Vemos estadisticos de la edad para decidir el reemplazo de valores perdidos
titanic$Age <- as.integer(titanic$Age)
summary(titanic$Age, na.rm = TRUE)
# reemplazamos por la media
titanic$Age[is.na(titanic$Age)] <- mean(titanic$Age, na.rm = TRUE)

# vemos los del embarque
titanic$Embarked <- as.factor(titanic$Embarked)
summary(titanic$Embarked, na.rm = TRUE)
#reemplazamos por el valor que mas aparece, S
titanic$Embarked[is.na(titanic$Embarked)] <- "S"

# revisamos los valores perdidos
sapply(titanic, function(x){sum(is.na(x))})


##########################################
## 3.2 Outliers ##########################
##########################################

# Asisnamos correctamente las clases a todas laas veriables
titanic$Age <- as.numeric(titanic$Age  )
titanic$Fare <- as.numeric(titanic$Fare  )
titanic$Pclass <- as.factor(titanic$Pclass  )
titanic$Sex <- as.factor(titanic$Sex  )
titanic$SibSp <- as.factor(titanic$SibSp  )
titanic$Parch <- as.factor(titanic$Parch  )
titanic$Embarked <- as.factor(titanic$Embarked  )
titanic$Survived <- as.factor(titanic$Survived  )


# Graficamos todas las variables para ver si hay ooutliers
plot(titanic$Survived)
plot(titanic$Pclass)
plot(titanic$Sex)
boxplot(titanic$Age)
plot(titanic$SibSp)
plot(titanic$Parch)
boxplot(titanic$Fare)
plot(titanic$Embarked)

#Exportación de los datos 
write.csv(titanic, "titanic.csv")


##########################################
## 4. Análisis de los datos ##############
##########################################

## 4.1 Selección de los grupos 
# Agrupación de variables de interes
titanic$ClaseBaja <- ifelse(titanic$Pclass == 3, TRUE, FALSE)

##########################################
## 4.2 Normalidad y homocedasticidad #####
##########################################

#Test de Normalidad
lillie.test(x = titanic$Age)
lillie.test(x = titanic$Fare)

ggplot(titanic,aes(x = titanic$Fare)) + 
  geom_histogram(aes(y = ..density..)) +
  theme_bw() 

#Test homocedasticidad
fligner.test(Age ~  Survived, data = titanic)
fligner.test(Fare ~ Survived, data = titanic)

##########################################
## 4.3 Pruebas estadísticas ##############
##########################################


##########################################
## 4.3.1 Correlaciones ###################
##########################################
correlaciones <- titanic[, c(6,10)]
chart.Correlation(correlaciones, histogram = TRUE, method = "pearson")


##########################################
## 4.3.2 Contraste de hipotesis ############
##########################################
# Precio del billete de los que sobreviven
sobrevive    <- titanic %>% filter(Survived == 1) %>% pull(Fare)
# Precio del billete de los que no sobreviven
nosobrevive <- titanic %>% filter(Survived == 0) %>% pull(Fare)
# El test de Wilcoxon 
wilcox.test(x=sobrevive,y=nosobrevive, paired = F)
mean(sobrevive)
mean(nosobrevive)
mean(sobrevive) - mean(nosobrevive)


##########################################
## 4.3.3 Regresion logistica ############
##########################################
# Creamos indices para dividir la base en entrenamiento y test con una misma relacion de la variable objetivo
# entre ellas y una division de 80-20
train_index <- createDataPartition(y = titanic$Survived, p = 0.8, list = FALSE, times = 1)
dat_train <- titanic[train_index, ]
dat_test  <- titanic[-train_index, ]
train_index <- sample(1:nrow(titanic), 0.8*nrow(titanic))  
dat_train <- titanic[train_index, ] 
dat_test  <- titanic[-train_index, ]

# Creamos el modelo logic
modelo1<-glm(Survived ~ ClaseBaja +  Age + Fare, data=dat_train, na.action = "na.omit", family = "binomial")
# Vemos los resultados
summary(modelo1)
#Vemos su acierto en el train
predicciones1 <- ifelse(test = modelo1$fitted.values > 0.5, yes = 1, no = 0)
matriz_confusion1 <- table(modelo1$model$Survived, predicciones1,
                          dnn = c("observaciones", "predicciones"))
matriz_confusion1
(matriz_confusion1[1,1] + matriz_confusion1[2,2])/sum(matriz_confusion1)


# Creamos el modelo2 logic y repetimos lo anterior
modelo2<-glm(Survived ~ ClaseBaja, data=dat_train, na.action = "na.omit", family = "binomial")
summary(modelo2)
predicciones2 <- ifelse(test = modelo2$fitted.values > 0.5, yes = 1, no = 0)
matriz_confusion2 <- table(modelo2$model$Survived, predicciones2,
                          dnn = c("observaciones", "predicciones"))
matriz_confusion2
(matriz_confusion2[1,1] + matriz_confusion2[2,2])/sum(matriz_confusion2)


#Aplicamos el mejor modelo al test
dat_test$Prediccionnumerica <- predict(modelo1, dat_test)
dat_test$PrediccionDico <- ifelse(dat_test$Prediccionnumerica > 0.5, 1, 0)
matriz_confusion <- table(dat_test$Survived, dat_test$PrediccionDico,
                          dnn = c("observaciones", "predicciones"))
matriz_confusion
(matriz_confusion[1,1] + matriz_confusion[2,2])/sum(matriz_confusion)
