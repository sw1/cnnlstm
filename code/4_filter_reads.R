#!/usr/bin/env Rscript

library(dada2)
library(ggplot2)
library(gridExtra)

seq_dir <- '~/earth/data/fastq'
filt_dir <- '~/earth/data/filtered'

dir.create(filt_dir,showWarnings=FALSE,recursive=TRUE)

seqs <- list.files(seq_dir,full.names=TRUE)
seqs_filt <- gsub('\\.fastq\\.gz$','_filt\\.fastq\\.gz',seqs)
seqs_filt <- gsub('\\/fastq\\/','\\/filtered\\/',seqs_filt)

samp <- sample(seqs,1)
plotQualityProfile(samp) +
  geom_vline(xintercept=c(10,100,125,150),linetype=3) +
  geom_hline(yintercept=c(25,30),linetype=3)


out <- filterAndTrim(seqs, seqs_filt,
                     trimLeft=c(10),truncLen=c(125),
                     maxN=0, maxEE=c(2), truncQ=2, rm.phix=TRUE,
                     compress=TRUE, multithread=60, verbose=TRUE)

saveRDS(out,'~/earth/data/filter_results.rds')
