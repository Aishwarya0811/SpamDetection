#read file
file.df<- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", sep=",", header = FALSE)
data = read.delim("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names", sep=" ", header=FALSE)
one<-as.factor(data[-(1:30),1])
colnames(file.df) = one
colnames(file.df)[58]<- "Type.Spam"
#question-1
#Normalise+10Predictors
normalise<-(sapply(file.df[,-58],scale))
norm<-cbind(normalise,file.df$Type.Spam)
colnames(norm)[58]<- "Type.Spam"
spam.predict<-data.frame(norm)
y<-abs(diff(as.matrix(aggregate(.~Type.Spam,spam.predict, mean))))
#Top 10 predictors
predictors<-head(sort(y[,-1],decreasing=TRUE),10)

#question-2
analyse.df<-spam.predict[,c("word_freq_your.","word_freq_000.","word_freq_remove.","char_freq_...4","word_freq_you.","word_freq_free." ,"word_freq_business.","word_freq_hp.","capital_run_length_total.","word_freq_our.","Type.Spam")]

#Partition Dataset
library(caret)
set.seed(13)
train.index <- createDataPartition(analyse.df$Type.Spam, p = 0.8, list = FALSE)
train.df <- analyse.df[train.index, ]
valid.df <- analyse.df[-train.index, ]

#LDA
install.packages("MASS")
library(MASS)
linear_model<-lda(Type.Spam ~., data = train.df)
#Predict 
pred2 <- predict(linear_model, valid.df)
#"class", "posterior", "x"
names(pred2)

#question-3

# Confusion Matrix
table(pred2$class, valid.df$Type.Spam)
sum(pred2$posterior[, 1] >=.5)

# lift chart decile chart 
install.packages("gains")
library(gains)
gain <- gains(valid.df$Type.Spam, pred2$x)

#Lift Chart
plot(c(0,gain$cume.pct.of.total*sum(valid.df$Type.Spam))~c(0,gain$cume.obs), 
     xlab = "# cases", ylab = "Cumulative", main = "", type = "l")
lines(c(0,sum(valid.df$Type.Spam))~c(0, dim(valid.df)[1]), lty = 5)

### Plot decile-wise chart
heights <- gain$mean.resp/mean(valid.df$Type.Spam)
decile_lift <- barplot(heights, names.arg = gain$depth,  ylim = c(0,3), col = "gold3",  
                       xlab = "Percentile", ylab = "Mean Response", 
                       main = "Decile-wise lift chart")

#Explanation :-
#confusion matrix
Accuracy <- ((531+231)/920)*100 
Sensitivity<- (231/(231+125))*100 #sensitivity <- (Actual spam identified as spam/ Actual spam)*100
#What can we learn from this matrix?
#There are two possible predicted classes: "Non Spam or o" and "Spam or 1". If we were predicting spam emails, for example, "1" would mean they are spam emails, and "0" would mean they are non-spam emails.
#The classifier made a total of 920 predictions .
#Out of those 920 , the classifier predicted "Spam" 356 times, and "Non Spam" 564 times.
#In reality, 264 are spam emails , and 656 non spam . The model is correctly classifying 231 email messages as SPAM out of actual 356 spam email messages(64.88%)

#DECILE LIFT CHARTS
# The top decile shows that our Model is  2.3 times approx better in identifying important class(Spam in our case) than naive or benchamrk or average when 10% is used.As  decile chart is telling us that , model is slightly more than twice likely to identify the class of interest as compared to average would do.
# Second decile which is detecting spam 2.5 times is more than first decile indicating that  model could have performed better. Further on descending order indicate that our model is doing a great job. 

#LIFT CHART 
# The model is accuarte in identifying approx 175 as Spam out of 200 email maessages whereas naive selection would have identified only 75 as Spam.
# Greater the area between lift curve and baseline, better the model indicating most gain 
# Lift chart shows improvement of the model i.e,how good is our model in predicting the Class of interest in Type.Spams .
# lift chart simply shows "how does my model compare to random guessing given X number of Email Messages".



