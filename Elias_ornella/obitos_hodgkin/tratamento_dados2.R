library(dplyr) 

# HODGKIN

setwd("D:/ufes/lestat/obitos_hodgkin/obitos_hodgkin/dados/hodgkin")

# menor 1 ano

d_hodg_menor1 <- readr::read_delim("obitos_hodg_menor1ano_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)

d_hodg_menor1 <- d_hodg_menor1 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  mutate(faixa_et = "menor_1ano") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() 
  


# 1 a 4 anos

d_hodg_1_4 <- readr::read_delim("obitos_hodg_1-4anos_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)


d_hodg_1_4 <- d_hodg_1_4 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  mutate(faixa_et = "1-4") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() 


# 5 a 9 anos

d_hodg_5_9 <- readr::read_delim("obitos_hodg_5-9anos_96-20.csv", ",",
                                escape_double = FALSE, 
                                trim_ws = TRUE)


d_hodg_5_9 <- d_hodg_5_9 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  mutate(faixa_et = "5-9") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() 

# 10 a 14 anos

d_hodg_10_14 <- readr::read_delim("obitos_hodg_10-14anos_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)


d_hodg_10_14 <- d_hodg_10_14 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  mutate(faixa_et = "10-14") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() 

# 15 a 19 anos

d_hodg_15_19 <- readr::read_delim("obitos_hodg_15-19anos_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)


d_hodg_15_19 <- d_hodg_15_19 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  mutate(faixa_et = "15-19") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() 


full_join(d_hodg_1_4,d_hodg_10_14)
# NEOPLASIAS

setwd("~/obitos_hodgking/dados/neoplasias")

# menor 1 ano

d_neop_menor1 <- readr::read_delim("obitos_neoplasias_menor1ano_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)

d_neop_menor1 <- d_neop_menor1 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  na_if("-")

# 1 a 4 anos

d_neop_1_4 <- readr::read_delim("obitos_neoplasias_1-4anos_96-20.csv", ",",
                                escape_double = FALSE, 
                                trim_ws = TRUE)

d_neop_1_4 <- d_neop_1_4 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  na_if("-")

# 5 a 9 anos

d_neop_5_9 <- readr::read_delim("obitos_neoplasias_5-9anos_96-20.csv", ",",
                                escape_double = FALSE, 
                                trim_ws = TRUE)

d_neop_5_9 <- d_neop_5_9 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  na_if("-")

# 10 a 14 anos

d_neop_10_14 <- readr::read_delim("obitos_neoplasias_10-14anos_96-20.csv", ",",
                                  escape_double = FALSE, 
                                  trim_ws = TRUE)

d_neop_10_14 <- d_neop_10_14 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  na_if("-")

# 15 a 19 anos

d_neop_15_19 <- readr::read_delim("obitos_neoplasias_15-19anos_96-20.csv", ",",
                                  escape_double = FALSE, 
                                  trim_ws = TRUE)

d_neop_15_19 <- d_neop_15_19 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-120:-129), ] %>% 
  na_if("-")

# POPULACAO

setwd("D:/ufes/lestat/obitos_hodgkin/obitos_hodgkin/dados/populacao")

# menor 1 ano

d_pop_menor1 <- readr::read_delim("populacao_menor1ano_00-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)

d_pop_menor1 <- d_pop_menor1 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-121:-123), ] %>% 
  na_if("-") 
# 1 a 4 anos

d_pop_1_4 <- readr::read_delim("populacao_1-4anos_00-20.csv", ",",
                               escape_double = FALSE, 
                               trim_ws = TRUE)

d_pop_1_4 <- d_pop_1_4 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-121:-123), ] %>% 
  na_if("-")

# 5 a 9 anos

d_pop_5_9 <- readr::read_delim("populacao_5-9anos_00-20.csv", ",",
                               escape_double = FALSE, 
                               trim_ws = TRUE)

d_pop_5_9 <- d_pop_5_9 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-121:-123), ] %>% 
  na_if("-")

# 10 a 14 anos

d_pop_10_14 <- readr::read_delim("populacao_10-14anos_00-20.csv", ",",
                                 escape_double = FALSE, 
                                 trim_ws = TRUE)

d_pop_10_14 <- d_pop_10_14 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-121:-123), ] %>% 
  na_if("-")

# 15 a 19 anos

d_pop_15_19 <- readr::read_delim("populacao_15-19anos_00-20.csv", ",",
                                 escape_double = FALSE, 
                                 trim_ws = TRUE)

d_pop_15_19 <- d_pop_15_19 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6)) %>% 
  .[c(-121:-123), ] %>% 
  na_if("-")
