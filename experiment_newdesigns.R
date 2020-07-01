library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(gridExtra)
library(reshape2)
library(magrittr)
library(zoo)

# sample data -------------------------------------------------------------------------------------
set.seed(44)
n = 10000
df_plot <- data.frame("class_0" = rnorm(n, 33, 5),
                      "class_1" = rnorm(n, 25, 3)) %>%
           melt( measure.vars = c("class_0", "class_1"))
  

# plot
ggplot(data=df_plot) + geom_histogram(aes(x=value, fill=variable), alpha=.4, bins=25, position = 'identity')


df_plot %>% arrange(desc(value)) %>%
  mutate(b2=diff(as.zoo(value), lag=1, na.pad=T)) %>%
  mutate(indx = as.numeric(row.names(.))) %>%
  
  ggplot() +
  geom_line(aes(x=indx, y=b2)) +
  scale_y_continuous(limits = c(0.1,-0.1))




# ssss -------------------------------------------------------------------------------------

# create bins
df_plot <- df_plot %>% 
  mutate(bined_values = cut(df_plot$value, breaks = 25) )

# tables from data
count_tbl <- table(df_plot$variable, df_plot$bined_values)
prob_tbl <- prop.table(count_tbl)

# tables with expected values if there was indepenedence
cout_exp_tbl <- chisq.test(table(df_plot$variable, df_plot$bined_values))$expected
prob_exp_tbl <- prop.table(cout_exp_tbl)

# wrangle data into better format
probabilites_binned <- data.frame(prob_tbl) %>% mutate(type="data") %>%
  set_colnames(c("class", "bin", "p", "type")) %>%
  rbind(., prob_exp_tbl %>% melt() %>%
          mutate(type="expected") %>% set_colnames(c("class", "bin", "p", "type"))) %>%
  group_by(bin, type) %>% summarise(p_0 = (p / sum(p))[1],
                                    p_1 = (p / sum(p))[2]) %>%
  
  mutate(logs_ent_0 = -1*log2(p_0)) %>%
  mutate(logs_ent_1 = -1*log2(p_1))
  



# plot
ggplot(data=probabilites_binned) +
  geom_bar(aes(x=bin, y=logs_ent_0, fill=type), alpha=.4, position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(hjust = 1, angle = 45))
    















