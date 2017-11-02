#!/usr/bin/env Rscript

library(tidyverse)

download_file <- function(file_name,url,overwrite=FALSE,retry=3){

  if (file.exists(file_name)){
    if (overwrite){
      message(sprintf('Overwriting %s',file_name))
    }else{
      message(sprintf('File exists: %s',file_name))
    }
  }

  retry <- 0

  while (TRUE){

    if (retry == 3) return(url)

    request <- RCurl::getURL(url,nobody=1L,header=1L)
    file_size <- as.numeric(gsub('^.*Length: ([0-9]+).*$','\\1',request))

    download.file(url,file_name,quiet=TRUE)

    if (file.size(file_name) == file_size) return(NULL)

    retry <- retry + 1

  }

}



DATA_DIR <- '~/earth/data'
FQ_DIR <- file.path(DATA_DIR,'fastq')
dir.create(FQ_DIR,recursive=TRUE,showWarnings=FALSE)

prj_id <- list.files(DATA_DIR)[grepl('PRJ',list.files(DATA_DIR))]

MAP <- read_delim(file.path(DATA_DIR,prj_id),delim='\t')
ERR <- gsub('\\.fastq\\.gz$','',unique(MAP$run_accession))
ERR_DL <- gsub('\\.fastq\\.gz$','',list.files(FQ_DIR))

MAP_TARG <-  MAP[!(ERR %in% ERR_DL),]

failures <- NULL
for (i in 1:nrow(MAP_TARG)){

  err <- MAP_TARG$run_accession[i]
  urls <- paste0('ftp://',strsplit(MAP_TARG$fastq_ftp[i],';')[[1]])

  file_names <- file.path(FQ_DIR,gsub('^.*(ERR[0-9]+.fastq.gz$)','\\1',urls))

  cat(sprintf('Downloading %s.\n',err))

  for (i in seq_along(urls)){
    attempt <- download_file(file_names[i],urls[i],overwrite=TRUE)
    if (!is.null(attempt)) cat('*** Failure ***\n')
    failures <- c(failures,attempt)
  }

}

cat(failures,sep='\n',file=file.path(DATA_DIR,'download_failures.txt'))
