rm(list=ls(all=TRUE))


###########################
#### Leitura dos Dados ####
###########################

library(readxl)
library(tidyverse)

menor_1 <- read_excel("C:/Users/micae/Downloads/Dados_Ivan/menor_1.xlsx", 
                      col_types = c("text", "numeric", "numeric", 
                                    "text"))

um_quatro <- read_excel("C:/Users/micae/Downloads/Dados_Ivan/um_quatro.xlsx", 
                        col_types = c("text", "numeric", "numeric", 
                                      "text"))

cinco_nove <- read_excel("C:/Users/micae/Downloads/Dados_Ivan/cinco_nove.xlsx", 
                         col_types = c("text", "numeric", "numeric", 
                                       "text"))

dez_quatorze <- read_excel("C:/Users/micae/Downloads/Dados_Ivan/dez_quatorze.xlsx", 
                           col_types = c("text", "numeric", "numeric", 
                                         "text"))

quinze_dezenove <- read_excel("C:/Users/micae/Downloads/Dados_Ivan/quinze_dezenove.xlsx", 
                              col_types = c("text", "numeric", "numeric", 
                                            "text"))


##############################################################################

###########################
#### Juntando os Dados ####
###########################

dados2<- rbind(menor_1, um_quatro, cinco_nove, dez_quatorze, quinze_dezenove)

dados2 <- na.exclude(dados2) #Exclui NA


###Separa por região

dados2 <- dados2 %>%
  separate(`Macrorregião de Saúde`, c("regiao", "macrorregiao")) %>%
  mutate(
    regiao = case_when(
      regiao = 1101 & regiao <= 1702 ~ "Norte",
      regiao = 2109 & regiao <= 2918 ~ "Nordeste",
      regiao = 3101 & regiao <= 3534 ~ "Sudeste",
      regiao = 4105 & regiao <= 4314 ~ "Sul",
      regiao = 5005 & regiao <= 5305 ~ "Centro-Oeste")) 


### Agrupa por ano
dados_agrupados <- dados2 %>%
  group_by(ano) %>%
  summarise(casos_soma = sum(casos))

ggplot(dados_agrupados)+
  geom_line(aes(dados_agrupados$ano, dados_agrupados$casos_soma))+
  xlab("") + 
  ylab("Casos de óbito por leucemia")


#############

###Agrupa por faixa_etaria por ano
dados_agrupados1 <- dados2 %>%
  group_by(faixa_etaria, ano) %>%
  summarise(casos_soma = sum(casos))


ggplot(dados_agrupados1)+
  geom_line(aes(dados_agrupados1$ano, dados_agrupados1$casos_soma,col=faixa_etaria))+
  xlab("") + 
  ylab("Casos de óbito por leucemia")


#Tabela de quantidade de casos por faixa etária

dados2 %>%
  group_by(faixa_etaria) %>%
  summarise(casos_soma = sum(casos))%>%
  mutate(Percentual = (casos_soma/sum(casos_soma))*100)


#Tabela de quantidade de casos por faixa etária e região

dados2 %>%
  group_by(faixa_etaria, regiao) %>%
  summarise(casos_soma = sum(casos))%>%
  mutate(Percentual = (casos_soma/sum(casos_soma))*100)


####################

###Agrupa por regiao
dados_agrupados2 <- dados2 %>%
  group_by(regiao, ano) %>%
  summarise(casos_soma = sum(casos))


ggplot(dados_agrupados2)+
  geom_line(aes(dados_agrupados2$ano, dados_agrupados2$casos_soma,col=regiao))+
  xlab("") + 
  ylab("Casos de óbito por leucemia") 
  

#########################

#Mapa de casos regiao

library(sf)
library(viridis)

dados_agrupados3 <- dados2 %>%
  group_by(regiao) %>%
  summarise(casos_soma = sum(casos))

shp1 <- st_read("regioes_2010.shp")

mapa <- merge(shp1, dados_agrupados3, by.x = "nome", by.y = "regiao", all = TRUE)


ggplot(mapa) + geom_sf(aes(fill = dados_agrupados3$casos_soma)) +
  scale_fill_viridis(alpha=0.80, direction = -1) +
  hrbrthemes::theme_ipsum() +labs(fill = "Casos de óbtido por leucemia")


###################

#Tabela de quantidade de casos por regiao

dados2 %>%
  group_by(regiao) %>%
  summarise(casos_soma = sum(casos))%>%
  mutate(Percentual = (casos_soma/sum(casos_soma))*100)

#########################################
# Aqui começa a descritiva com as taxas
# de óbtido a cada 100 mil habitantes
##############################################

###Importação da base de população para cálculo das taxas de óbito

populacao_lestat <- read_excel("C:/Users/micae/Downloads/Dados_Ivan/populacao_lestat.xlsx", 
                                col_types = c("numeric", "numeric", "numeric", 
                                              "numeric", "numeric", "numeric", 
                                               "numeric", "numeric", "numeric", 
                                                "numeric", "numeric"))
#View(populacao_lestat)


populacao_lestat<- subset(populacao_lestat, select = -c(3:10))


populacao_lestat <- populacao_lestat %>%
  mutate(
    cod_regiao = case_when(
      cod_regiao = 1101 & cod_regiao <= 1702 ~ "Norte",
      cod_regiao = 2109 & cod_regiao <= 2918 ~ "Nordeste",
      cod_regiao = 3101 & cod_regiao <= 3534 ~ "Sudeste",
      cod_regiao = 4105 & cod_regiao <= 4314 ~ "Sul",
      cod_regiao = 5005 & cod_regiao <= 5305 ~ "Centro-Oeste"))

#View(populacao_lestat)

###Agrupa a população da regiao em cada ano

#pop_regiao_ano <- populacao_lestat %>%
  #group_by(cod_regiao,ano) %>%
  #summarise(pop = sum(pop))


###Agrupa a população por região somando todos os anos

pop_regiao <- populacao_lestat %>%
  group_by(cod_regiao) %>%
  summarise(pop = sum(pop))


casos_regiao <- dados2 %>%
  group_by(regiao) %>%
  summarise(casos = sum(casos))

taxa_reg<- cbind(pop_regiao, casos_regiao)%>%
  subset(select = -c(3))

taxa_reg<- mutate(taxa_reg, taxa = (taxa_reg$casos/taxa_reg$pop)*100000)


#Tabela da taxa por óbtido a cada 100 mil habitantes

taxa_reg %>% subset(select = -c(2,3))



##Mapa de taxa dos casos


mapa2 <- merge(shp1, taxa_reg, by.x = "nome", by.y = "cod_regiao", all = TRUE)


ggplot(mapa2) + geom_sf(aes(fill = taxa_reg$taxa)) +
  scale_fill_viridis(alpha=0.80, direction = -1) +
  hrbrthemes::theme_ipsum() +labs(fill = "Taxa de óbito a cada 100 mil habitantes")


#####################################

############

###Agrupa a população em cada ano

pop_ano <- populacao_lestat %>%
  group_by(ano) %>%
  summarise(pop = sum(pop))


#####################################

##Média geometrica população por ano

a<-1
for(i in 1:22){
  
  a<- a*pop_ano$pop[i]  
  
}

med_geo<- a^{1/22}
med_geo #64787273

#####################

#Estimando as população de 2018, 2019 e 2020


for(i in 23:25){
  
    pop_ano[i,1]<- pop_ano[i-1,1]+1
    
    pop_ano[i,2]<- pop_ano[i-1,2]*(1+med_geo)
    
}

casos_pop<- cbind(dados_agrupados, pop_ano)

casos_pop<- subset(casos_pop, select = -c(3))


casos_pop<-mutate(casos_pop, (casos_pop$casos_soma/casos_pop$pop)*100000)%>%
            rename(taxa = "(casos_pop$casos_soma/casos_pop$pop) * 100000")

#View(casos_pop)



### Taxa de óbito a cada 1000 mil habitante

ggplot(casos_pop)+
  geom_line(aes(casos_pop$ano, casos_pop$taxa))+
  xlab("") + 
  ylab("Taxa de óbito a cada 1000 mil habitantes")
