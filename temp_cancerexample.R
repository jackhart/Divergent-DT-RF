library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(gridExtra)
library(reshape2)
library("kdensity")


datasets <- read.csv("/home/jack/Documents/Divergent_DT_RF/temp.csv")



# decision tree function
find_minimum_entropy <- function(df){

    # default values
  threshold <- NA
  best_entropy_left <- 1000
  best_entropy_right <- 1000
  best_average_entropy <- 1000
  
  # order data by X
  df_ordered <- df %>% arrange(X)
  
  for(i in 1:(nrow(df_ordered)-1)){
    # subset data
    left_y <- df_ordered$Y[1:i]
    right_y <- df_ordered$Y[(i+1):nrow(df_ordered)]
    
    # find class probabilites
    probabilities_left <-prop.table(table(left_y))
    probabilities_right <-prop.table(table(right_y))
    
    prob_x_left <- i / nrow(df_ordered)
    prob_x_right <- (nrow(df_ordered) - i) / nrow(df_ordered)
    
    # calculate conditional entropy
    
    
    entropy_left <- prob_x_left * (-1*sum(probabilities_left*log2(probabilities_left)))
    if (is.na(entropy_left)){
      entropy_left = 0
    }
    entropy_right <- prob_x_right * (-1*sum(probabilities_right*log2(probabilities_right)))
    if (is.na(entropy_right)){
      entropy_right = 0
    }
    entropy_split = entropy_left + entropy_right
    # weighted average of entropy
    #  entropy_split = ( (length(left_y) * entropy_left) + (length(right_y) * entropy_right) ) / (length(df_ordered$Y) )
    
    left_ent_lst <<- c(left_ent_lst, entropy_left)
    right_ent_lst <<- c(right_ent_lst, entropy_right)
    total_ent_lst <<- c(total_ent_lst, entropy_split)
    
    if(entropy_split < best_average_entropy){
      threshold <- df_ordered$X[i]
      best_entropy_left <- entropy_left
      best_entropy_right <- entropy_right
      best_average_entropy <- entropy_split
    }
  }
  return(c(threshold, best_entropy_left, best_entropy_right,best_average_entropy))
}


setwd("/home/jack/Documents/Divergent_DT_RF")
getwd()

breast_cancer <- read.csv("UCI/data/data.csv")


coul <- colorRampPalette(brewer.pal(4, "Spectral") )(8)[c(6,2)]
breast_cancer %>%
ggplot() + geom_density(aes(x=radius_mean, fill = as.factor(diagnosis)), color="grey90", alpha = .4) +
  geom_density(aes(x=radius_mean), color = "grey30", size=1.5)  +
  theme_minimal() + scale_fill_manual(label=c("Benign", "Malignant"), values = coul) +
  labs(title="Class Distributions of Mean Tumor Radius", fill="")



# decision threshold of a classic decision tree 

# global variables because I'm lazy
left_ent_lst <<- c()
right_ent_lst <<- c()
total_ent_lst <<- c()


tree_vals <- breast_cancer %>% dplyr::select(diagnosis, radius_mean) %>% 
                rename(Y=diagnosis, X=radius_mean) %>% find_minimum_entropy()

g <- breast_cancer %>%
  ggplot() + geom_density(aes(x=radius_mean, fill = as.factor(diagnosis)), color="grey90", alpha = .4) +
  geom_density(aes(x=radius_mean), color = "grey30", size=1.5)  +
  theme_minimal() + scale_fill_manual(label=c("Benign", "Malignant"), values = coul) +
  labs(title="Class Distributions of Mean Tumor Radius", fill="", x="Mean Radius", y="Density") 
  geom_vline(aes(xintercept=tree_vals[1]))




# 0 cancer
# 1 benign
# lambda is benign

# determine lambda as proportion in class 0 (i.e. benign)
lambda = sum(breast_cancer$diagnosis == "B") / nrow(breast_cancer)

# sufficient statisitcs
mu_b <- mean(breast_cancer$perimeter_mean[breast_cancer$diagnosis == "B"])
sd_b <- sd(breast_cancer$perimeter_mean[breast_cancer$diagnosis == "B"])
mu_c <- mean(breast_cancer$perimeter_mean[breast_cancer$diagnosis == "M"])
sd_c <- sd(breast_cancer$perimeter_mean[breast_cancer$diagnosis == "M"])

# calculate lambda divergence
densities_p <- dnorm(breast_cancer$perimeter_mean %>% sort(), mean=mu_b, sd=sd_b)
densities_c <- dnorm(breast_cancer$perimeter_mean  %>% sort(), mean=mu_c, sd=sd_c)

D_l1 = densities_p * log2(densities_p / (lambda * densities_p+ (1-lambda)*densities_c))
D_l0 = densities_c * log2(densities_c / (lambda * densities_p + (1-lambda)*densities_c))
D_lall = lambda * D_l1 + (1-lambda) * D_l0


# possible convex version
convex_p <- -log2(densities_p)
convex_c <- -log2(densities_c)


# -- 
D_l1 = convex_p * log2(convex_p / (lambda * convex_p+ (1-lambda)*convex_c))
D_l0 = convex_c * log2(convex_c / (lambda * convex_p + (1-lambda)*convex_c))
D_lall = lambda * D_l1 + (1-lambda) * D_l0


# ?   -- 
plot(convex_p, convex_c, D_l1)



# plot of simplified class distributions
df_plot <- data.frame("benign_p" = densities_p,
           "cancer_p" = densities_c,
           "x" = breast_cancer$radius_mean %>% sort(),
           #"entropy_benign" = densities_p*log2(1/densities_p),
           #"entropy_cancer" = densities_c*log2(1/densities_c)) %>% 
           "entropy_benign" = log2(1/densities_p),
           "entropy_cancer" = log2(1/densities_c)
           #"test" = densities_p / densities_c,
           #"div" = log2(densities_p / densities_c) 
           ) %>% 
          #mutate(entropy_pq = entropy_benign / entropy_cancer) %>%
          mutate(entropy_pq = log2(densities_p / densities_c)) %>%
          #mutate(entropy_pq = exp(sqrt((entropy_benign - entropy_cancer)**2) )) %>%
          mutate_if(is.numeric, ~replace(., is.na(.), 0)) 

df_plot %>%
          select(-c("entropy_cancer", "entropy_benign")) %>%
  melt( measure.vars = c("benign_p", "cancer_p", "entropy_pq" )) %>%
  ggplot() +
  geom_line(aes(x=x, y=value, color=variable)) +
  scale_y_continuous(breaks = NULL) +
  scale_y_continuous(limits = c(0,1)) 

# plot that minimizes adjusted relative entropy
ggplot(data = df_plot) + geom_line(aes(x=x, y=sqrt((entropy_pq - 1)**2)) )
df_plot$x[which.min(sqrt((df_plot$entropy_pq - 1)**2))]


#
# Relative Entropy -- can be used to determine minimum value per variable
# Total Relative Entopy gives average difference in entropy -- pick value with max?
#

sd_1 <- sd(breast_cancer$perimeter_mean)
#mu_1 <-mean(breast_cancer$perimeter_mean)

lr = 1.06 * sd_1 * nrow(breast_cancer)**(-1/5)


# find class probabilites
probabilities_left <-prop.table(table(breast_cancer$diagnosis))
probabilities_right <-prop.table(table(right_y))

prob_x_left <- i / nrow(df_ordered)
prob_x_right <- (nrow(df_ordered) - i) / nrow(df_ordered)


# calculate conditional entropy
entropy_left <- prob_x_left * (-1*sum(probabilities_left*log2(probabilities_left)))
entropy_right <- prob_x_right * (-1*sum(probabilities_right*log2(probabilities_right)))

base_entropy = entropy_left + entropy_right

# choose random x
# calculate binary cross entropy = prev entropy

for(i in 1:100){

  # 
  
  
  
}







temp <- data.frame("benign_p" = densities_p,
           "cancer_p" = densities_c,
           "x" = breast_cancer$radius_mean %>% sort(),
           #"entropy_benign" = densities_p*log2(1/densities_p),
           #"entropy_cancer" = densities_c*log2(1/densities_c)) %>% 
           #"entropy_benign" = log2(1/densities_p),
           #"entropy_cancer" = log2(1/densities_c),
           "test" = densities_p / densities_c,
           "div" = log2(densities_p / densities_c) ) %>% 
  filter(div >= 0, div <=1)
  


# plot of simplified class distributions
data.frame("benign_p" = densities_p,
           "cancer_p" = densities_c,
           "x" = breast_cancer$radius_mean %>% sort()) %>% 
  melt( measure.vars = c("benign_p", "cancer_p")) %>%
  ggplot() +
  geom_line(aes(x=x, y=value, color=variable)) +
  scale_y_continuous(limits = c(0,0.25)) +
  geom_vline(xintercept = mean(temp$x))




entrop_b <- 0.5 * log2(2 * pi * exp(1) * sd_b^2 )
entrop_c <- 0.5 * log2(2 * pi * exp(1) * sd_c^2 )


kde_b = kdensity(breast_cancer$radius_mean[breast_cancer$diagnosis == "B"],, start = "gumbel", kernel = "gaussian")
kde_c = kdensity(breast_cancer$radius_mean[breast_cancer$diagnosis == "M"],, start = "gumbel", kernel = "gaussian")


benign_p = kde_b(breast_cancer$radius_mean)
cancer_p = kde_c(breast_cancer$radius_mean)

data.frame("benign_p" = benign_p,
           "cancer_p" = cancer_p,
           "x" = breast_cancer$radius_mean) %>% 
  melt( measure.vars = c("benign_p", "cancer_p")) %>%
  ggplot() +
  geom_line(aes(x=x, y=value, color=variable))



calc_divergence <- function(p1,p2) {
  return(p1*log2(p1 / p2))
}

#divergences <- outer(benign_p, cancer_p, calc_divergence)

divgence_to_p <- mapply(calc_divergence, benign_p, cancer_p)


data.frame("D" = divgence_to_p,
           "p" = benign_p) %>% 
  arrange(p) %>%
  ggplot() +
  geom_line(aes(x=p, y=D))


benign_p[which(divgence_to_p == min(divgence_to_p))]


#plot
library(plot3D)
persp3D(benign_p, cancer_p, divergences,  phi = 30, theta = 45, col = "white", shade = 0.5, main="Surface Plot")

scatter3D(kernel_regression_2$x, kernel_regression_2$y, kernel_regression_2$z, phi = 30, col = "grey30",
          pch = 20, cex = .7, main="Scatter Plot")





D_l1 = benign_p * log2(benign_p / (lambda * benign_p+ (1-lambda)*cancer_p))
D_l0 = cancer_p * log2(cancer_p / (lambda * benign_p + (1-lambda)*cancer_p))

D_lall = lambda * D_l1 + (1-lambda) * D_l0



g <- breast_cancer %>%
  ggplot() + geom_density(aes(x=radius_mean, fill = as.factor(diagnosis)), color="grey90", alpha = .4) +
  geom_line(data = data.frame("x"=d_cancer$x, "y"=d_cancer$y), aes(x=x, y=y), color=coul[2], size=1.5) +
  geom_line(data = data.frame("x"=d_benign$x, "y"=d_benign$y), aes(x=x, y=y), color=coul[1], size=1.5) +
  theme_minimal() + scale_fill_manual(label=c("Benign", "Malignant"), values = coul) +
  geom_line(data = data.frame("x"=breast_cancer$radius_mean, "y"=D_lall), aes(x=x, y=y), size=1.5, color="purple") +
  geom_vline(aes(xintercept=radius_mean[which(D_lall == min(D_lall))]), size=1.2) +
  labs(title="Lambda Divergence Threshold", fill="", x="Mean Radius", y="Density")

  

g <- breast_cancer %>%
  ggplot() + geom_density(aes(x=radius_mean, fill = as.factor(diagnosis)), color="grey90", alpha = .4) +
  geom_line(data = data.frame("x"=d_cancer$x, "y"=d_cancer$y), aes(x=x, y=y), color=coul[2], size=1.5) +
  geom_line(data = data.frame("x"=d_benign$x, "y"=d_benign$y), aes(x=x, y=y), color=coul[1], size=1.5) +
  theme_minimal() + scale_fill_manual(label=c("Benign", "Malignant"), values = coul) +
  geom_vline(aes(xintercept=tree_vals[1]), size=1.2) +
  labs(title="Classic Decision Tree Threshold", fill="", x="Mean Radius", y="Density")
  

png(filename="overallplot.png", type="cairo", units="in", bg = "transparent",
     width=6,  height=5, pointsize=12, res=400)
g

dev.off()




# Next we will use lambda divergence to determine the best split

bins = seq(0,16,by=.4)
discrete_1_x <- cut(df_ordered$X[df_ordered$Y == 1], breaks=bins, right = FALSE)
dicrete_0_x <- cut(df_ordered$X[df_ordered$Y == 0], breaks=bins, right = FALSE)

probs_1 <- unname(prop.table(table(discrete_1_x)))
probs_0 <- unname(prop.table(table(dicrete_0_x)))

# calculate the divergence
x_vals <- data.frame("X" = bins[-1],
                     'prob_1' = probs_1,
                     'prob_0' = probs_0) %>% dplyr::select(X, prob_1.Freq, prob_0.Freq)

D_KL <- x_vals$prob_1.Freq * log2( x_vals$prob_1.Freq / ( (x_vals$prob_1.Freq + x_vals$prob_0.Freq) / 2) )
D_KL[is.na(D_KL)] <- 0.0 # there are some divide by zero errors
D_KL2 <- x_vals$prob_0.Freq * log2( x_vals$prob_0.Freq / ( (x_vals$prob_1.Freq + x_vals$prob_0.Freq) / 2) )
D_KL2[is.na(D_KL2)] <- 0.0

JS_div <- 0.5 * D_KL + 0.5 * D_KL2

# optimal bin selected
names(prop.table(table(discrete_1_x)))[23]


g <- breast_cancer %>%
  ggplot() + geom_density(aes(x=radius_mean, fill = as.factor(diagnosis)), color="grey90", alpha = .4) +
  geom_line(data = data.frame("x"=d_cancer$x, "y"=d_cancer$y), aes(x=x, y=y), color=coul[2], size=1.5) +
  geom_line(data = data.frame("x"=d_benign$x, "y"=d_benign$y), aes(x=x, y=y), color=coul[1], size=1.5) +
  theme_minimal() + scale_fill_manual(label=c("Benign", "Malignant"), values = coul) +
  geom_vline(aes(xintercept=tree_vals[1]), size=1.2) +
  labs(title="Classic Decision Tree Threshold", fill="", x="Mean Radius", y="Density")






png(filename="xor.png", type="cairo", units="in", bg = "transparent",
    width=6,  height=5, pointsize=12, res=400)

datasets %>% dplyr::select(xor_x_1, xor_x_2, xor_y) %>%
  ggplot() + geom_point(aes(x=xor_x_1, y=xor_x_2, color=as.factor(xor_y)), size=2) +
  theme_minimal() + scale_color_manual(label=c("Class 0", "Class 1"), values = coul) +
  labs(title="XOR", fill="", x= "X_1", y="X_2", color="") + theme(text = element_text(size=16))
  
dev.off()




png(filename="donut.png", type="cairo", units="in", bg = "transparent",
    width=6,  height=5, pointsize=12, res=400)

datasets %>% dplyr::select(donut_x_1, donut_x_2, donut_y) %>%
  ggplot() + geom_point(aes(x=donut_x_1, y=donut_x_2, color=as.factor(donut_y)), size=2) +
  theme_minimal() + scale_color_manual(label=c("Class 0", "Class 1"), values = coul) +
  labs(title="Donut", fill="", x= "X_1", y="X_2", color="") + theme(text = element_text(size=16))

dev.off()



png(filename="moons.png", type="cairo", units="in", bg = "transparent",
    width=6,  height=5, pointsize=12, res=400)

datasets %>% dplyr::select(moons_x_1, moons_x_2, moons_y) %>%
  ggplot() + geom_point(aes(x=moons_x_1, y=moons_x_2, color=as.factor(moons_y)), size=2) +
  theme_minimal() + scale_color_manual(label=c("Class 0", "Class 1"), values = coul) +
  labs(title="Moons", fill="", x= "X_1", y="X_2", color="") + theme(text = element_text(size=16))

dev.off()



data_in <- read.csv("/home/jack/Documents/Divergent_DT_RF/results/total_results.csv")

df <- data_in %>% group_by(model, dataset) %>% 
  summarise(median_train_acc = median(train_accuracies),
            median_test_acc = median(test_accuracies),
            mean_train_acc = mean(train_accuracies),
            mean_test_acc = mean(test_accuracies),
            test_sd = sd(test_accuracies),
            train_sd = sd(train_accuracies),) %>% ungroup() # %>% write.csv("/home/jack/Documents/Kernel_Implementations_Random_Forests/results/temp.csv")



t.test(df$mean_test_acc[df$model == "KeDT"], alternative = "two.sided", mu = df$mean_test_acc[df$model == "classic"])


prop.test(mean_train_acc)



data_in %>% group_by(model, dataset) %>%
  summarise(med_train_times = median(train_times)) %>%
  ungroup() %>% spread(model, med_train_times) %>%
  mutate(perc_dif = (classic - KeDT) / classic)
  


data_in <- read.csv("/home/jack/Documents/Divergent_DT_RF/results/total_results_n.csv")


data_in %>% filter(model == "KeDT") %>%
  group_by(n) %>%
  summarise(med_time = median(train_times)) %>%
  
  ggplot() + geom_line(aes(x=n, y=med_time))











