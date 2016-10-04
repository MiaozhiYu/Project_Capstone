setwd('F:/Miaozhi/Academic/Data_Science/Bootcamp/Project_Capstone/nycdsa-capstone')
data = read.csv('./data/cs-training.csv', header =T)
data1 = read.csv('./data/cs-training-outlier-f10.csv', header =T)
data = data[ ,-1]
data1 = data1[ ,-1]
library(VIM)
library(mice)
library(adabag)
library(corrplot)
library(caret)
#library(doMC)
library(neuralnet)
library(nnet)
library(devtools)
library(pROC)
library(clusterGeneration)
library(NeuralNetTools)
library(mice)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(psych)
library(ggpubr)
library(reshape2)
library(randomForest)
library(Amelia)
###Data Summary####
summary(data)
summary(data1)
str(data)
####Plot missing value####
missmap(data)
aggr(data,prop = F, numbers = T)
md.pattern(data)
matrixplot(data, interactive = F)

ggplot_missing <- function(x){
  
  x %>% 
    is.na %>%
    melt %>%
    ggplot(data = .,
           aes(x = X2,
               y = X1)) +
    geom_raster(aes(fill = value)) +
    scale_fill_grey(name = "",
                    labels = c("Present","Missing")) +
    theme_minimal() + 
    theme(axis.text.x  = element_text(angle=45, vjust=0.5)) + 
    labs(x = "Variables in Dataset",
         y = "Rows / observations")
}

ggplot_missing(data)


####Correltion plot####
corrplot(cor(data[complete.cases(data),]), method="circle", type = "lower", order="hclust",
         col=colorRampPalette(brewer.pal(11,"Spectral"))(8))
cor(data[complete.cases(data),])

corrplot(cor(data1[complete.cases(data1),]), method="circle", type = "lower", order="hclust",
         col=colorRampPalette(brewer.pal(11,"Spectral"))(8))
cor(data1[complete.cases(data1),])
####Heat map for correlation####
qplot(x=Var1, y=Var2, data=melt(cor(data[complete.cases(data),], use="p")), fill=value, geom="tile") +
  scale_fill_gradient2(limits=c(-1, 1))
qplot(x=Var1, y=Var2, data=melt(cor(data1[complete.cases(data1),], use="p")), fill=value, geom="tile") +
  scale_fill_gradient2(limits=c(-1, 1))
####PCA#####
fa.parallel(data, #The data in question.
            fa = "pc", #Display the eigenvalues for PCA.
            n.iter = 100) #Number of simulated analyses to perform.
abline(h = 1) #Adding a horizontal line at 1.
pc_bodies = principal(data, #The data in question.
                      nfactors = 5, #The number of PCs to extract.
                      rotate = "none")
pc_bodies

heatmap <- qplot(x=Var1, y=Var2, data=melt(cor(data)), geom="tile",
                 fill=value)

fa.parallel(data1, #The data in question.
            fa = "pc", #Display the eigenvalues for PCA.
            n.iter = 100) #Number of simulated analyses to perform.
abline(h = 1) #Adding a horizontal line at 1.
pc_bodies = principal(data1, #The data in question.
                      nfactors = 4, #The number of PCs to extract.
                      rotate = "none")
pc_bodies
pca <- prcomp(data, scale=T,na.action = na.omit())
melted <- cbind(variable.group, melt(pca$rotation[,1:5]))

barplot <- ggplot(data=data) +
  geom_bar(aes(x=Var1, y=value, fill=variable.group), stat="identity") +
  facet_wrap(~Var2)

####Important variables####
fit=randomForest(factor(SeriousDlqin2yrs)~., data=data[complete.cases(data),])
varImp(fit)
varImpPlot(fit,type=2)
