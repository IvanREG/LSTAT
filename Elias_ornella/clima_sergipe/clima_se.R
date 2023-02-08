library(dplyr)
library(tidyr)
library(tidyverse)
library(lubridate)
library(ggstatsplot)


nordeste <- read.csv("northeast.csv")
sudeste <- read.csv("southeast.csv")
sergipe <- nordeste |> filter(state=="SE")

sergipe <- sergipe |> 
  janitor::clean_names() %>% 
  select(-index)

glimpse(sergipe)

sergipe <- sergipe %>% 
  mutate(across(names(sergipe[, 3:19]), ~ case_when(.x == -9999 ~ NA_real_,
                                                  TRUE ~ as.numeric(.x))))
sergipe <- sergipe %>% 
  unite('time', c('data', 'hora'), sep = ' ', remove = TRUE) %>% 
  mutate(time = ymd_hms(time))

sergipe <- sergipe |> 
  select(time,latitude,longitude,height,precipitacao_total_horario_mm,radiacao_global_kj_m,
         umidade_relativa_do_ar_horaria,pressao_atmosferica_ao_nivel_da_estacao_horaria_m_b,
         temperatura_do_ar_bulbo_seco_horaria_c,vento_velocidade_horaria_m_s,station) |> 
  rename(prcp=precipitacao_total_horario_mm,
         radiacao=radiacao_global_kj_m,
         umidade=umidade_relativa_do_ar_horaria,
         press_atm=pressao_atmosferica_ao_nivel_da_estacao_horaria_m_b,
         temp = temperatura_do_ar_bulbo_seco_horaria_c,
         vento_velo=vento_velocidade_horaria_m_s)





 