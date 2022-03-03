### Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicação Mobile

# Este código foi criado para o projeto da Formação Cientista de Dados da
# Data Science Academy

# Problema de Negócio: construir um modelo de aprendizado de máquina para 
# determinar se um clique é fraudulento ou não.

# Devido ao tamanho do dataset, as análises deste projeto foram realizadas 
# utilizando apenas o dataset train_sample, que contém 100.000 amostras aleatórias 
# do dataset principal

# As informações foram disponibilizado pela empresa TalkingData e podem ser
# encontradas no Kaggle
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

setwd("~/Documentos/DSA/FCD/BigDataRAzure/Cap20/Projetos-1-2/Projeto-1")

# Carregando Bibliotecas

library (inspectdf)
library (tidyr)
library (readr)
library (dplyr)
library (ggplot2)
library (Amelia)
library (ROSE)
library (caTools)
library(randomForest)
library(e1071)
library(caret)
library(caTools)
library(rmarkdown)
library(GGally)
library(ggscatmat)
install.packages('ggscatmat')

# Importando o dataset
df <- read_csv("train_sample.csv")

str(df)

## PREPARAÇÃO DOS DADOS

# Criando as colunas de dia, hora e período do dia
df$day <- format(df$click_time,"%d")
df$hour <- format(df$click_time,"%H")
df$min <- format(df$click_time, "%M")
df$sec <- format(df$click_time, "%S")

df$shift <- format(df$click_time,"%H")

shift_day <-  function(i){
  if (i >= 0 & i < 6)
    return ("MADRUGADA")
  else if (i >= 6 & i <= 12)
    return ("MANHA")
  else if (i > 12 & i <= 18)
    return ("TARDE")
  else if (i > 18 & i <= 23)
    return ("NOITE")
}

df$shift <- sapply(as.integer(df$shift), shift_day)

# Transformando as variáveis para o tipo Fator
df$is_attributed <- as.factor(df$is_attributed)
df$shift <- as.factor(df$shift)

## ANÁLISE EXPLORATÓRIA

# Verificando dados NA no dataset
missmap(df,
        main = "Mapa de Dados Missing", 
        col = c("blue", "black"), 
        legend = FALSE)

# Verificando dados NA na variável attributed_time
# A variável attributed_time possui 99.77% de dados NA (os dados NA = não houve download)
# A informação referente a esta variável já pode ser obtida através da nossa variável target

sapply(df, function(x) sum(is.na(x)/length(x))*100)

# Há menos acesso à noite e no dia 06 quando comparados 
# com os outros períodos. A taxa de acessos durante a manhã e é
# mais do que a soma dos acessos da noite e tarde.
prop <- table(df$shift, useNA = "always")
prop <- prop.table(prop) * 100
round(prop, digits = 1)

prop <- table(df$day, useNA = "always")
prop <- prop.table(prop) * 100
round(prop, digits = 1)

# Mas quando verificamos a quantidade de horas de acesso para cada um dos 4 dias
# explica-se quantidade de downloads menor do dia 06. 
df %>% 
  ggplot(aes(x = hour, y = day, color = factor(is_attributed))) +
  geom_point(aes(color = factor(is_attributed)))


# As dados do dataset forma recolhidos de segunda a quinta-feira
df$weekday <- weekdays(df$click_time)
df %>% 
  ggplot(aes(x = hour, y = weekday, color = factor(is_attributed))) +
  geom_point(aes(color = factor(is_attributed)))

# Verificando a distribuição de todas as features observamos que 
# apesar de algumas variáveis serem classificadas como numéricas, elas tem  
# comportamento de fator
df %>%
  select(-attributed_time)%>%
  mutate_all(as.factor) %>%
  inspect_cat() %>% 
  show_plot()

# A visão geral da distribuição de alguns dados do dataset

df %>% 
  ggplot(aes(x=reorder(shift, shift, function(x)-length(x)))) +
  geom_bar(fill='steelblue') +
  labs(x='shift') 

df %>% 
  ggplot(aes(x=reorder(hour, hour, function(x)-length(x)))) +
  geom_bar(fill='steelblue') +
  labs(x='hour') 

df %>% 
  ggplot(aes(x=reorder(day, day, function(x)-length(x)))) +
  geom_bar(fill='steelblue') +
  labs(x='day')

df %>% 
  ggplot(aes(x=reorder(os, os, function(x)-length(x)))) +
  geom_bar(fill='steelblue') +
  labs(x='os') 

df %>% 
  ggplot(aes(x=reorder(channel, channel, function(x)-length(x)))) +
  geom_bar(fill='steelblue') +
  labs(x='channel')


# Visualizando a distribuição da variável Target, podemos observar que os dados 
# não estão balanceadas com 99.77% dos dados rotulados como 0 (o que faz sentido,
# uma vez que há menos transações fraudulentas do que não fraudulentas)
df %>% 
  count(is_attributed) %>% 
  ggplot(aes(x=rev(is_attributed), y=n, fill=is_attributed))+
  geom_bar(stat = "identity")+
  scale_fill_brewer(palette="Set1")+
  theme(legend.position = "bottom")+
  labs(y="N° observacoes", x = "")

prop <- table(df$is_attributed)
prop <- prop.table(prop) * 100
round(prop, digits = 1)

############# SPLIT E BALANCEAMENTO DOS DADOS ###########
# Split dos dados
df <- df %>% select (-attributed_time, -click_time)
split = sample.split(df$is_attributed, SplitRatio = 0.70)

# Datasets de treino e de teste
trainset = subset(df, split == TRUE)
testset = subset(df, split == FALSE)

# Balanceando os dados de treino
trainset.balanced <- ovun.sample(is_attributed ~ ., data=trainset,
                                 N=nrow(trainset), p=0.5, 
                                 seed=1, method="both")$data

table(trainset.balanced$is_attributed)

# Balanceando os dados de teste
#trainset_df <- trainset %>% select(-attributed_time, -click_time, -day, -hour, -shift)
#str(trainset_df)
testset.balanced <- ovun.sample(is_attributed ~ ., data=testset,
                                N=nrow(testset), p=0.5, 
                                seed=1, method="both")$data

table(testset.balanced$is_attributed)


# Gerando o modelo com os dados treino
model_glm <- glm(formula = is_attributed ~ . , family = binomial(link='logit'), 
                 data = trainset.balanced)

# Resumo
summary(model_glm)

# Fazendo as previsoes com os dados de teste
fitted.probabilities <- predict(model_glm, newdata = testset.balanced, type = 'response')

# Calculando os resultados
fitted.results <- ifelse(fitted.probabilities > 0.5, 1, 0)

# Conseguimos quase 65% de acurácia
misClasificError <- mean(fitted.results != testset.balanced$is_attributed)
print(paste('Acuracia', 1-misClasificError))

# Criando a confusion matrix
table(testset.balanced$is_attributed, fitted.probabilities > 0.5)

############### OTIMIZANDO - SELECIONANDO VARIÁVEIS COM RANDOM FOREST #####################

## TREINANDO O MODELO

# Aplicando o Random Forest para selecionar variáveis: channel, app, device, ip 
# foram as variáveis selecionadas
model_rf <- randomForest(is_attributed ~ .,
                         data = df, 
                         ntree = 100, nodesize = 10, importance = T)

varImpPlot(model_rf)


# Criando dataset com as variáveis que serão usadas
df_new <- df %>% select(is_attributed, channel, app, ip, device, os)
str(df_new)
# Fazendo Split dos dados do novo dataset
split <- sample.split(df_new$is_attributed, SplitRatio = 0.70)

# Datasets de treino e de teste
trainset <- subset(df_new, split == TRUE)
testset <- subset(df_new, split == FALSE)

# Balanceando os dados de treino
trainset.balanced <- ovun.sample(is_attributed ~ ., data=trainset,
                                 N=nrow(trainset), p=0.5, 
                                 seed=1, method="both")$data

table(trainset.balanced$is_attributed)

# Balanceando os dados de teste
testset.balanced <- ovun.sample(is_attributed ~ ., data=testset,
                                N=nrow(testset), p=0.5, 
                                seed=1, method="both")$data

table(testset.balanced$is_attributed)


# Treinando o modelo 
model_glm_v2 <- glm(formula = is_attributed ~ channel + app + device, 
                    family = binomial(link = 'logit'), 
                    data = trainset)

# Podemos ver que as variáveis ip, app e channel sao as variaveis mais significativas
summary(model_glm_v2)

# Fazendo as previsoes com os dados de teste
pred_test_glm2 <- predict(model_glm_v2, newdata = testset.balanced, type = 'response')

# Calculando os resultados
pred_test_glm_results <- ifelse(pred_test_glm2 > 0.5, 1, 0)

# Conseguimos 50% de acurácia o que está bem abaixo do modelo anterior
misClasificError <- mean(pred_test_glm_results != testset.balanced$is_attributed)
print(paste('Acuracia', 1-misClasificError))

# Criando a confusion matrix
table(testset.balanced$is_attributed, pred_test_glm2 > 0.5)

#############

# Treinando o modelo SVM a acurácia chegou em 77%
model_svm <- svm(is_attributed ~ channel + app + device, 
                 data = trainset.balanced, 
                 type = 'C-classification', 
                 kernel = 'radial') 

# Previsões

# Previsões nos dados de treino
pred_train_svm <- predict(model_svm, trainset.balanced) 
mean(pred_train_svm == trainset.balanced$is_attributed)  

# Previsões nos dados de teste
pred_test_svm <- predict(model_svm, testset.balanced) 
mean(pred_test_svm == testset.balanced$is_attributed)  

# Matriz de Confusão
cm_svm <- confusionMatrix(table(pred = pred_test_svm, testset.balanced$is_attributed), 
                          positive = "1")

# Métricas do Modelo
acuracia_svm <- cm_svm$overall["Accuracy"]
precisao_svm <- cm_svm$byClass["Precision"]
recall_svm <- cm_svm$byClass["Recall"]
f1_svm <- cm_svm$byClass["F1"]
metric_svm <- c(acuracia_svm, precisao_svm, recall_svm, f1_svm)


## Modelo NaiveBayes - apresentou acurácia de 74,7%

# Treinando o Modelo
model_nb <- naiveBayes(is_attributed ~ channel + app + device, 
                      data = trainset.balanced)

# Previsões com dados treino
pred_train_nb <- predict(model_nb, trainset.balanced)
mean(pred_train_nb == trainset.balanced$is_attributed)  

# Prevendo com dados teste
pred_test_nb <- predict(model_nb, testset.balanced)
mean(pred_test_nb == testset.balanced$is_attributed) 

# Matriz de Confusão
mc_nb <- confusionMatrix(table(pred = pred_test_nb, testset.balanced$is_attributed), 
                         positive = "1")

# Métricas do Modelo
acuracia_nb <- mc_nb$overall["Accuracy"]
precisao_nb <- mc_nb$byClass["Precision"]
recall_nb <- mc_nb$byClass["Recall"]
f1_nb <- mc_nb$byClass["F1"]
metric_nb <- c(acuracia_nb, precisao_nb, recall_nb, f1_nb)


## Modelo Random Forest - apresentou acurácia de 90% mas apresentou a melhor 
# precisão entre os modelos treinados (97,9%)

# Treinando o Modelo
model_forest <- randomForest(is_attributed ~ channel + app + device,
                             data = trainset.balanced, 
                             ntree = 100, nodesize = 10)

# Previsões
pred_train_forest <- predict(model_forest, trainset.balanced)
mean(pred_train_forest == trainset.balanced$is_attributed)

pred_test_forest <- predict(model_forest, testset.balanced)
mean(pred_test_forest == testset.balanced$is_attributed)

# Matriz de Confusão
mc_forest <- confusionMatrix(table(pred = pred_test_forest, 
                                   testset.balanced$is_attributed), positive = "1")

# Métricas do Modelo
acuracia_forest <- mc_forest$overall["Accuracy"]
precisao_forest <- mc_forest$byClass["Precision"]
recall_forest <- mc_forest$byClass["Recall"]
f1_forest <- mc_forest$byClass["F1"]
metric_forest <- c(acuracia_forest, precisao_forest, recall_forest, f1_forest)


## AVALIAÇÃO DO MODELO
# Comparando as métricas dos modelos: A acurácia do modelos não chegaram num patamar
# esperado, logo, será usada nova técnica para balanceamento dos dados
metric_svm 
metric_nb
metric_forest

# Gerando a curva ROC do modelo com melhor acurácia
library(ROCR)

pred <- prediction(as.numeric(pred_test_forest), testset.balanced$is_attributed)
roc <- performance(pred, "tpr","fpr")

plot(roc, col = rainbow(10), main = "Curva ROC")
abline(a=0, b=1)

# Gerando a Curva de Precision/Recall
perf <- performance(pred, "prec", "rec")
plot(perf, main = "Curva Precision/Recall")











####### NÃO


# Treinando o modelo sem balancear os dados - Acurácia: 99,77%
# Apesar da otima acurácia
#trainset_df <- trainset %>% select(-attributed_time, -click_time)
model_svm <- svm(is_attributed ~ channel+ app + device + ip + os + min, 
                 data = trainset, 
                 type = 'C-classification', 
                 kernel = 'radial') 

# Previsões com dados de teste
pred_test <- predict(model_svm, testset) 

# Percentual de previsões corretas com dataset de teste
mean(pred_test == testset$is_attributed)  

# Matriz de Confusão
cm_svm <- confusionMatrix(table(pred = pred_test, testset$is_attributed), 
                          positive = "1")

# Métricas do Modelo
acuracia_svm <- cm_svm$overall["Accuracy"]
precisao_svm <- cm_svm$byClass["Precision"]
recall_svm <- cm_svm$byClass["Recall"]
f1_svm <- cm_svm$byClass["F1"]
metric_svm <- c(acuracia_svm, precisao_svm, recall_svm, f1_svm)

# Modelo 2 - modelo SVM otimizado com variáveis selecionadas
# O modelo foi testado com diferentes variáveis preditoras e o que apresentou
# a acurácia de 84% e na precisão. Por ainda não apresentar a acurácia esperada segui
# para outro algoritmo

# Treinando o modelo
model_svm2 <- svm(is_attributed ~ ip + channel + app, 
                  data = trainset.balanced, 
                  type = 'C-classification', 
                  kernel = 'radial') 
?svm

# Previsões nos dados de treino
pred_train2 <- predict(model_svm2, trainset.balanced) 
mean(pred_train2 == trainset.balanced$is_attributed)  

# Previsões nos dados de teste
pred_test2 <- predict(model_svm2, testset.balanced) 
mean(pred_test2 == testset.balanced$is_attributed)  

# Matriz de Confusão
mc_svm2 <- confusionMatrix(table(pred = pred_test2, testset.balanced$is_attributed), 
                           positive = "1")

# Métricas do Modelo
acuracia_svm2 <- mc_svm2$overall["Accuracy"]
precisao_svm2 <- mc_svm2$byClass["Precision"]
recall_svm2 <- mc_svm2$byClass["Recall"]
f1_svm2 <- mc_svm2$byClass["F1"]
metric_svm2 <- c(acuracia_svm2, precisao_svm2, recall_svm2, f1_svm2)

## Arvore: Apresenta acurácia de 91%, e ótima precisão, sendo a melhor entre 
# os modelos : app + channel + ip

library(rpart)
library(rpart.plot)

pruneControl = rpart.control(minsplit = 15, minbucket = 5)
model_tree = rpart(is_attributed ~ channel+ app + device + ip + os, 
                   data = trainset.balanced, control = pruneControl)

# Visualização da árvore
prp(model_tree)

#pred_tree_train <- predict(model_tree, trainset.balanced)
#pred_tree_train <- as.data.frame(pred_tree_train)
#pred_tree_train["classe"] <- ifelse(pred_tree_train >= 0.5, 1, 0)

pred_tree_test <- predict(model_tree, testset.balanced)
pred_tree_test <- as.data.frame(pred_tree_test)
pred_tree_test["classe"] <- ifelse(pred_tree_test >= 0.5, 1, 0)

mc_tree <- confusionMatrix(as.factor(pred_tree_test$class[,2]), 
                           as.factor(pred_tree_test[,6]), positive = "1", 
                           mode = "prec_recall")
mc_tree$table

acuracia_tree <- mc_tree$overall["Accuracy"]
precisao_tree <- mc_tree$byClass["Precision"]
recall_tree <- mc_tree$byClass["Recall"]
f1_tree <- mc_tree$byClass["F1"]
metric_tree <- c(acuracia_tree, precisao_tree, recall_tree, f1_tree)
metric_tree
