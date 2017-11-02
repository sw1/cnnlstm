#!/usr/bin/env Rscript

library(tidyverse)

DATA_DIR <- '~/earth/data'
META_DIR <- file.path(DATA_DIR,'metadata')
dir.create(META_DIR,recursive=TRUE,showWarnings=FALSE)

prj_id <- list.files(DATA_DIR)[grepl('PRJ',list.files(DATA_DIR))]
MAP <- read_delim(file.path(DATA_DIR,prj_id),delim='\t')

ERS <- unique(MAP$secondary_sample_accession)
ERS <- ERS[!(ERS %in% gsub('\\.xml','',list.files(META_DIR)))]

url <- 'http://www.ebi.ac.uk/ena/data/view/%s&display=xml'

for (ers in ERS) download.file(sprintf(url,ers),file.path(META_DIR,paste0(ers,'.xml')))
