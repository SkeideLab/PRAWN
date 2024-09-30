import_timeseries <- function(file) {
  #read_csv(file, col_types = cols()) %>%
  #  filter(!is.na(Ozone))
  #  
  dfslid <- read.csv(file)
  dfslid <- t(dfslid)
  times <- rep(seq(-0.4, 1.0, length.out = 351), 3) # assuming 3 conditions
  # name the columns from -0.4 to 1.0 with 351 steps
  colnames(dfslid) <- times
  
  dfslid
}

sign_flip <- function(data, n_perm = 10000) {
  # debug
  #n_perm=1024*8
  #data <- tar_read(tr_data_con, branches=1)
  #times <- rep(seq(-0.4, 1.0, length.out = 351),3) # assuming 1 condition
  
  # implement permutations
  times <- seq(-0.4, 1.0, length.out = 351) # assuming 1 condition
  chance = 0.5
  
  data <- t(data)
  # name the columns from -0.4 to 1.0 with 351 steps
  colnames(data) <- times
  
  n_sub = dim(data)[1] # remove condition and tar_group TODO maybe
  n_tp = dim(data)[2]
  
  # remove chance from ts
  df0 = data - chance #[1:38,]
  
  # average time series to be used for statistics
  df0_avg = colMeans(df0)
  
  # empty array of n_perm x n_tp
  perm0_averages = array(NA, dim=c(n_perm, n_tp))
  
  for (i in 1:n_perm){
    # set seed
    set.seed(i)
    # random array of 1 and -1 of subject lenght
    perm = sample(c(1, -1), nrow(df0), replace = TRUE)
    #print(perm)
    # multiply the perm array with the data
    df0_perm = df0 * perm
    # add mean to the perm_averages
    perm0_averages[i,] = colMeans(df0_perm)
  }
  
  # merge df0_avg and perm0_averages
  X = rbind(df0_avg, perm0_averages) #+ chance
  X
  
}

