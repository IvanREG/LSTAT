library(dplyr)

# HODGKIN

setwd("D:/ufes/lestat/obitos_hodgkin/obitos_hodgkin/dados/hodgkin")

# menor 1 ano

d_hodg_menor12 <- readr::read_delim("obitos_hodg_menor1ano_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)

d_hodg_menor1 <- d_hodg_menor12 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>%   
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste"),
         `2000` = 0,
         `2003` = 0,
         `2004` = 0,
         `2005` = 0,
         `2006` = 0,
         `2007` = 0,
         `2008` = 0,
         `2010` = 0,
         `2011` = 0,
         `2012` = 0,
         `2013` = 0,
         `2014` = 0,
         `2015` = 0,
         `2016` = 0,
         `2017` = 0,
         `2019` = 0,
         `2020` = 0) %>% 
  .[c(-120:-129), ] %>% 
  select(-`1996`) %>% 
  mutate(faixa_et = "menor_1ano") %>%
  mutate(base="hodgkin") %>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)

# 1 a 4 anos

d_hodg_1_4 <- readr::read_delim("obitos_hodg_1-4anos_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)


d_hodg_1_4 <- d_hodg_1_4 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste"),
         `2005` = 0,
         `2007` = 0,
         `2008` = 0,
         `2009` = 0,
         `2010` = 0,
         `2011` = 0,
         `2012` = 0,
         `2013` = 0,
         `2014` = 0,
         `2015` = 0,
         `2016` = 0,
         `2017` = 0,
         `2019` = 0) %>% 
  .[c(-120:-129), ]%>% 
  select(-c(`1996`, `1997`, `1998`, `1999`)) %>% 
  mutate_at(
    c(
      '2000',
      '2001',
      '2002',
      '2003',
      '2004',
      '2005',
      '2006',
      '2007',
      '2008',
      '2009',
      '2010',
      '2011',
      '2012',
      '2013',
      '2014',
      '2015',
      '2016',
      '2017',
      '2018',
      '2019',
      '2020'
    ),
    as.numeric
  ) %>%
  mutate(faixa_et = "1-4") %>%
  mutate(base = "hodgkin")

# 5 a 9 anos

d_hodg_5_9 <- readr::read_delim("obitos_hodg_5-9anos_96-20.csv", ",",
                                escape_double = FALSE, 
                                trim_ws = TRUE)


d_hodg_5_9 <- d_hodg_5_9 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-120:-129), ]%>%
  select(-c(`1996`, `1997`, `1998`, `1999`))%>% 
  mutate(faixa_et = "5-9") %>% 
  mutate(base="hodgkin")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)


# 10 a 14 anos

d_hodg_10_14 <- readr::read_delim("obitos_hodg_10-14anos_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)


d_hodg_10_14 <- d_hodg_10_14 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-120:-129), ]%>% 
  select(-c(`1996`, `1997`, `1998`, `1999`))%>% 
  mutate(faixa_et = "10-14") %>% 
  mutate(base="hodgkin")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)


# 15 a 19 anos

d_hodg_15_19 <- readr::read_delim("obitos_hodg_15-19anos_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)


d_hodg_15_19 <- d_hodg_15_19 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-120:-129), ]%>% 
  select(-c(`1996`, `1997`, `1998`, `1999`))%>% 
  mutate(faixa_et = "15-19") %>%
  mutate(base="hodgkin")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)



# NEOPLASIAS

setwd("D:/ufes/lestat/obitos_hodgkin/obitos_hodgkin/dados/neoplasias")

# menor 1 ano

d_neop_menor1 <- readr::read_delim("obitos_neoplasias_menor1ano_96-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)

d_neop_menor1 <- d_neop_menor1 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-120:-129), ]%>% 
  select(-c(`1996`, `1997`, `1998`, `1999`))%>% 
  mutate(faixa_et = "menor_1ano") %>% 
  mutate(base="neoplasias")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)


# 1 a 4 anos

d_neop_1_4 <- readr::read_delim("obitos_neoplasias_1-4anos_96-20.csv", ",",
                                escape_double = FALSE, 
                                trim_ws = TRUE)

d_neop_1_4 <- d_neop_1_4 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-120:-129), ]%>% 
  select(-c(`1996`, `1997`, `1998`, `1999`))%>% 
  mutate(faixa_et = "1-4") %>% 
  mutate(base="neoplasias")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)


# 5 a 9 anos

d_neop_5_9 <- readr::read_delim("obitos_neoplasias_5-9anos_96-20.csv", ",",
                                escape_double = FALSE, 
                                trim_ws = TRUE)

d_neop_5_9 <- d_neop_5_9 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-120:-129), ]%>% 
  select(-c(`1996`, `1997`, `1998`, `1999`))%>% 
  mutate(faixa_et = "5-9") %>%
  mutate(base="neoplasias")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)


# 10 a 14 anos

d_neop_10_14 <- readr::read_delim("obitos_neoplasias_10-14anos_96-20.csv", ",",
                                  escape_double = FALSE, 
                                  trim_ws = TRUE)

d_neop_10_14 <- d_neop_10_14 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-120:-129), ]%>% 
  select(-c(`1996`, `1997`, `1998`, `1999`))%>% 
  mutate(faixa_et = "10-14") %>%
  mutate(base="neoplasias")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)


# 15 a 19 anos

d_neop_15_19 <- readr::read_delim("obitos_neoplasias_15-19anos_96-20.csv", ",",
                                  escape_double = FALSE, 
                                  trim_ws = TRUE)

d_neop_15_19 <- d_neop_15_19 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-120:-129), ]%>% 
  select(-c(`1996`, `1997`, `1998`, `1999`))%>% 
  mutate(faixa_et = "15-19") %>% 
  mutate(base="neoplasias")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)



dados_cancer <- rbind(d_hodg_1_4,d_hodg_10_14,d_hodg_15_19,d_hodg_5_9,d_hodg_menor1,d_neop_menor1,d_neop_1_4,d_neop_10_14,d_neop_15_19)

saveRDS(dados_cancer,"D:/ufes/lestat/obitos_hodgkin/obitos_hodgkin/dados/dados_cancer.rds")

# POPULACAO

setwd("D:/ufes/lestat/obitos_hodgkin/obitos_hodgkin/dados/populacao")

# menor 1 ano

d_pop_menor1 <- readr::read_delim("populacao_menor1ano_00-20.csv", ",",
                                   escape_double = FALSE, 
                                   trim_ws = TRUE)

d_pop_menor1 <- d_pop_menor1 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-121:-123), ]%>% 
  mutate(faixa_et = "menor_1ano") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(Total = apply(.[,c(2:22)],1,function(x){sum(as.numeric(x))}))%>% 
  mutate(base="populacao")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)





# 1 a 4 anos

d_pop_1_4 <- readr::read_delim("populacao_1-4anos_00-20.csv", ",",
                               escape_double = FALSE, 
                               trim_ws = TRUE)

d_pop_1_4 <- d_pop_1_4 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-121:-123), ]%>% 
  mutate(faixa_et = "1-4") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(Total = apply(.[,c(2:22)],1,function(x){sum(as.numeric(x))}))%>% 
  mutate(base="populacao")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)




# 5 a 9 anos

d_pop_5_9 <- readr::read_delim("populacao_5-9anos_00-20.csv", ",",
                               escape_double = FALSE, 
                               trim_ws = TRUE)

d_pop_5_9 <- d_pop_5_9 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-121:-123), ]%>% 
  mutate(faixa_et = "5-9") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(Total = apply(.[,c(2:22)],1,function(x){sum(as.numeric(x))}))%>% 
  mutate(base="populacao")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)




# 10 a 14 anos

d_pop_10_14 <- readr::read_delim("populacao_10-14anos_00-20.csv", ",",
                                 escape_double = FALSE, 
                                 trim_ws = TRUE)

d_pop_10_14 <- d_pop_10_14 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-121:-123), ]%>% 
  mutate(faixa_et = "10-14") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(Total = apply(.[,c(2:22)],1,function(x){sum(as.numeric(x))}))%>% 
  mutate(base="populacao")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)




# 15 a 19 anos

d_pop_15_19 <- readr::read_delim("populacao_15-19anos_00-20.csv", ",",
                                 escape_double = FALSE, 
                                 trim_ws = TRUE)


d_pop_15_19 <- d_pop_15_19 %>% 
  rename(macro_saude = `Macrorregi�o de Sa�de`) %>% 
  mutate(codigo_macro = stringr::str_sub(macro_saude, end = 4),
         macro_saude = stringr::str_sub(macro_saude, start = 6),
         regiao = case_when(
           stringr::str_sub(codigo_macro, end = 2) == 11 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 12 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 13 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 14 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 15 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 16 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 17 ~ "norte",
           stringr::str_sub(codigo_macro, end = 2) == 21 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 22 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 23 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 24 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 25 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 26 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 27 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 28 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 29 ~ "nordeste",
           stringr::str_sub(codigo_macro, end = 2) == 31 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 32 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 33 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 35 ~ "sudeste",
           stringr::str_sub(codigo_macro, end = 2) == 41 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 42 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 43 ~ "sul",
           stringr::str_sub(codigo_macro, end = 2) == 51 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 52 ~ "centro-oeste",
           stringr::str_sub(codigo_macro, end = 2) == 53 ~ "centro-oeste")) %>% 
  .[c(-121:-123), ]%>% 
  mutate(faixa_et = "15-19") %>% 
  sapply(function(x){ifelse(x=="-",0,x)}) %>% 
  as.data.frame() %>% 
  mutate(Total = apply(.[,c(2:22)],1,function(x){sum(as.numeric(x))}))%>% 
  mutate(base="populacao")%>% 
  mutate_at(c('2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013', '2014','2015', '2016','2017','2018', '2019', '2020'), as.numeric)



dados_pop <- rbind(d_pop_1_4,d_pop_10_14,d_pop_15_19,d_pop_5_9,d_pop_menor1)

saveRDS(dados_pop,"D:/ufes/lestat/obitos_hodgkin/obitos_hodgkin/dados/dados_populacao.rds")

