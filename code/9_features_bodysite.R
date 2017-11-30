library(tidyverse)
library(viridis)
library(reticulate)
library(keras)
use_condaenv('tensorflow3')

f <- import_from_path(module='functions',path='/data/sw1/cnnlstm/code/features_functions.py',convert=FALSE)

visualize_model <- function(model,include_gradients=FALSE){

  back <- keras::backend()

  lstm_layer <- model$get_layer('biLSTM')
  output_layer <- model$get_layer('output')

  inputs <- list(model$layers[[1]]$input,
                 model$layers[[2]]$input,
                 model$layers[[3]]$input,
                 back$learning_phase())

  outputs <- vector(mode='list',length=2)
  outputs[1] <- model$outputs
  outputs[[2]] <- lstm_layer$output

  lstm_weights <- lstm_layer$get_weights()
  lstm_weights_forget_f <- lstm_weights[[1]][,33:64]
  lstm_weights_forget_b <- lstm_weights[[4]][,33:64]
  lstm_weights_forget <- list(lstm_weights_forget_f,lstm_weights_forget_b)

  if (include_gradients){
    loss <- back$mean(model$output)
    grads <- back$gradients(loss,lstm_layer$output)
    grads_norm <- grads/(back$sqrt(back$mean(back$square(grads))) + 1e-5)
    outputs <- c(outputs,grads_norm)
  }

  all_function <- back$Function(inputs,outputs)

  output_function <- back$Function(list(output_layer$input),model$outputs)

  out <- list(all_function=all_function,
              output_function=output_function,
              lstm_weights_forget=lstm_weights_forget)

  return(out)

}

get_read_idxs <- function(gen){

  read_len <- py_to_r(gen$read_len)
  n_reads <- py_to_r(gen$n_reads)
  n_pad <- py_to_r(gen$n_pad)

  unlist(lapply(seq(1,by=read_len + n_pad,length.out=n_reads),
                function(i) i:(i+read_len-1)))
}

get_batch_data <- function(gen,batch,read_idxs){

  n_batch <- py_to_r(gen$n_batch)
  read_len <- py_to_r(gen$read_len)
  n_reads <- py_to_r(gen$n_reads)
  kmer_dict <- py_to_r(gen$rev_key)

  tmp <- batch[[1]][[1]][,read_idxs,]

  batch_data <- matrix('',nrow=n_batch*n_reads*read_len,ncol=7,
                       dimnames=list(NULL,c('batch_idx','sample','seq_idx',
                                            'read_idx','kmer_idx','kmer','label')))

  position <- 1
  for (i in 1:n_batch){
    b <- tmp[i,,]
    samp_id <- batch[[3]][i]
    samp_label <- which.max(batch[[2]][i,]) - 1

    r <- 1

    for (j in 1:n_reads){
      for (k in 1:read_len){

        read <- b[r,]
        idx <- as.character(which.max(read) - 1)
        kmer <- kmer_dict[[idx]]

        batch_data[position,1] <- as.character(i) # batch idx
        batch_data[position,2] <- samp_id # sample id
        batch_data[position,3] <- as.character(r) # seq idx
        batch_data[position,4] <- batch[[4]][[samp_id]][j] # read idx
        batch_data[position,5] <- idx # kmer idx
        batch_data[position,6] <- kmer # kmer
        batch_data[position,7] <- samp_label # sample label

        position <- position + 1
        r <- r + 1

      }
    }
  }

  batch_data <- data.frame(batch_data,stringsAsFactors=FALSE)

  return(batch_data)

}

get_batch_labels <- function(batch_data,path='/data/sw1/bodysite/data/reads_labels'){
  fns <- list.files(path)
  ids <- gsub('.label','',fns)

  batch_data <- batch_data %>%
    filter(sample %in% ids)

  read_labels <- do.call(rbind,lapply(unique(batch_data$sample),
         function(x) data.frame(sample=x,
                                read_csv(sprintf('%s%s%s%s',path,'/',x,'.label'),col_names=FALSE),
                                stringsAsFactors=FALSE))) %>%
    rename(read_idx=X1,taxon=X2) %>%
    mutate(read_idx=as.character(read_idx)) %>%
    separate(taxon,into=paste0('T',1:12),sep=';')

  batch_data <- suppressWarnings({
    batch_data %>%
    left_join(read_labels,by=c('sample','read_idx'))
  })

  return(batch_data)

}


gen <- f$GenerateBatch(path='/data/sw1/bodysite/data/',one_hot=TRUE)
read_idxs <- get_read_idxs(gen)

val_gen <- py_to_r(gen$generate(dataset='val'))
batch <- iter_next(val_gen)
batch_data <- get_batch_data(gen,batch,read_idxs)
batch_data <- get_batch_labels(batch_data)





model <- load_model_hdf5('~/MiscData/cnn_lstm/bodysite_out/model_final_k4.hdf5')




vis <- visualize_model(model,include_gradients=TRUE)

out <- vis$all_function(c(batch[[1]],0))
out[[2]] <- out[[2]][,read_idxs,]

data.frame(pred=apply(out[[1]],1,which.max),truth=apply(batch[[2]],1,which.max))


n_batch <- dim(out[[2]])[1]
time_distributed_scores <- sapply(seq_len(n_batch),
                                  function(i) vis$output_function(list(out[[2]][i,,])))
time_distributed_scores <- do.call(rbind,time_distributed_scores)
tds_idx <- apply(time_distributed_scores,1,which.max)
time_distributed_scores[,1] <- time_distributed_scores[,1]*-1
time_distributed_scores <- sapply(seq_along(tds_idx),function(i) time_distributed_scores[i,tds_idx[i]])
tds <- data.frame(batch_idx = as.character(rep(1:n_batch,each=length(time_distributed_scores)/n_batch)),
                  seq_idx = as.character(1:(length(time_distributed_scores)/n_batch)),
                  w=time_distributed_scores)


bi <- 1
tmp <- batch_data %>%
  filter(batch_idx == bi,
         T7 %in% c('Bacteroidales','Lachnospiraceae')) %>%
  mutate(seq_idx=as.integer(seq_idx))


data.frame(out[[2]][bi,,]) %>%
  mutate(seq_idx=row_number()) %>%
  filter(seq_idx %in% unique(tmp$seq_idx)) %>%
  gather(node,w,-seq_idx) %>%
  mutate(node=gsub('X','',node)) %>%
  left_join(tmp,by='seq_idx') %>%
  filter(!is.na(T7)) %>%
  ggplot() +
  geom_raster(aes(x=node,y=seq_idx,fill=w)) +
  scale_fill_viridis() +
  facet_wrap(~T7)

tds %>%
  left_join(batch_data,by=c('batch_idx','seq_idx')) %>%
  group_by(T7) %>%
  mutate(n_taxa=n()) %>%
  ungroup() %>%
  arrange(seq_idx,label,batch_idx) %>%
  mutate(rank_taxa=dense_rank(desc(n_taxa)),
         taxa=ifelse(rank_taxa > 10,'Other',T7),
         batch_idx=as.integer(batch_idx),
         seq_idx=as.integer(seq_idx),
         batch_idx=factor(batch_idx,ordered=TRUE,levels=unique(batch_idx))) %>%
  ggplot() +
  geom_raster(aes(x=batch_idx,y=seq_idx,fill=w)) +
  scale_fill_viridis() +
  facet_wrap(~taxa) +
  theme_classic()
