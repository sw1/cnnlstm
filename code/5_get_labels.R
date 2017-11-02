#!/usr/bin/env Rscript

library(tidyverse)

DATA_DIR <- '~/earth/data'
META <- readRDS(file.path(DATA_DIR,'metadata.rds'))

prj_id <- list.files(DATA_DIR)[grepl('PRJ',list.files(DATA_DIR))]
MAP <- read_delim(file.path(DATA_DIR,prj_id),delim='\t')

MAP %>%
  select(PRIMARY_ID=secondary_sample_accession,SampleID=run_accession) %>%
  left_join(META,by='PRIMARY_ID') %>%
  filter(!(lake %in% c('Lake Haus'))) %>%
  select(SampleID,lake) %>%
  mutate(lake=as.integer(as.factor(lake))-1) %>%
  write_csv(file.path(DATA_DIR,'labels.csv'),col_names=FALSE)
