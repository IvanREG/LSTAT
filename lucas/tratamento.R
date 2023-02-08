library(tidyverse)
library(lubridate)

dados <- readr::read_csv('northeast.csv') %>% 
  janitor::clean_names() %>% 
  select(-index)

dados <- dados %>% 
  filter(year(data) > 2017,
         year(data) < 2021,
         state == 'AL')

glimpse(dados)

dados <- dados %>% 
  mutate(across(names(dados[, 3:19]), ~ case_when(.x == -9999 ~ NA_real_,
                                                  TRUE ~ as.numeric(.x))))

dados <- dados %>% 
  unite('time', c('data', 'hora'), sep = ' ', remove = TRUE) %>% 
  mutate(time = ymd_hms(time))

dados %>% 
  readr::write_csv('northeast-tratado.csv')
