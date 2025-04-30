# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline

# Load packages required to define the pipeline:
library(targets)
library(tarchetypes) # Load other packages as needed.

# Set target options:
tar_option_set(
  packages = c("dplyr", 
               "magrittr",
               "ggplot2", 
               "ggpubr",
               "grid", # to use rasterplot
               "png",
               "readr",
               "stringr", # R1
               "tidyr", # R1
               "tidyverse", # R1
               "broom", # R1
               "permuco") # packages that your targets use
  # format = "qs", # Optionally set the default storage format. qs is fast.
)

# Run the R scripts in the R/ folder with your custom functions:
tar_source()
# tar_source("other_functions.R") # Source other scripts as needed.

tar_option_set() # to be able to skip targets?

mycolors = c("#656364",
             "#332288", #  "#004949", #darkgreen
             "#882155"
)

# derived from https://www.jantau.com/post/sparplantag-reloaded/
# colorblind friendlyness tested using https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40-%23792427-%23D1BDA2-%2354828E
# it is from library(gameofthrones) (not compatible with this R version)
# https://github.com/aljrico/gameofthrones
# pal <- got(3, option = "Daenerys", direction = -1)
# or: scale_fill_got_d(option = "Daenerys", direction = -1)
# 
rkcolors  = c("#54828e",
              "#d1bda2",
              "#792427")

petroff_colors = c("#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd")

# Replace the target list below with your own:
list(

  tar_target(
    name = tr_file2,
    command = "../models/timeresolved.csv",
    format = "file",
    description = "timeresolved timeseries data file" 
  ),
  
  # tr
  tar_target(
    name = tr_data,
    command = read.csv(tr_file2),
    description = "timeresolved timeseries data" 
  ),
  # manual group by
  tar_target(name = tr_data_inter, command = tr_data %>% subset(species=="inter") %>% select(-c(species))),
  tar_target(name = tr_data_intra_human, command = tr_data %>% subset(species=="intra_human") %>% select(-c(species))),
  tar_target(name = tr_data_intra_monkey, command = tr_data %>% subset(species=="intra_monkey") %>% select(-c(species))),
  tar_target(name = tr_data_con, 
             command = list(tr_data_inter, tr_data_intra_human, tr_data_intra_monkey),
             iteration = "list"),
  
  # load the images for plotting
  tar_target(
    name = human_image_file,
    command = "../plots/stimuli/humans.png",
    format = "file",
    description = "human image file" 
  ),
  tar_target(
    name = monkey_image_file,
    command = "../plots/stimuli/monkeys.png",
    format = "file",
    description = "monkey image file" 
  ),
  tar_target(
    name = both_image_file,
    command = "../plots/stimuli/both.png",
    format = "file",
    description = "human and monkey image file" 
  ),
  tar_target(
    name = human_image,
    command = readPNG(human_image_file),
    description = "human image" 
  ),
  tar_target(
    name = monkey_image,
    command = readPNG(monkey_image_file),
    description = "monkey image" 
  ),
  tar_target(
    name = both_image,
    command = readPNG(both_image_file),
    description = "both image" 
  ),
  tar_target(name = stimulus_images, 
             command = list(both_image, human_image, monkey_image),
             iteration = "list"),
  
  
  # group by contrast
  #tar_group_by(
  #  name = tr_data_con, # contrast (inter, intra_human, intra_monkey) 
  #  tr_data, 
  #  species,
  #),    
  
  tar_target(
    name = tr_permutations,
    command = sign_flip(tr_data_con),
    pattern = map(tr_data_con),
    iteration="list",
    description = "timeresolved sign flipped permutation time series" 
  ),
  
  tar_target(
    name = clustermass,
    command = {
      cm <- compute_clustermass(tr_permutations, 
                                  threshold=0.01,
                                  aggr_FUN=sum, 
                                  alternative = "greater")
      cm$main[,"pvalue"]
    },
    pattern = map(tr_permutations),
    iteration="list",
    description = "cluster mass test" 
  ),
  tar_target(
    name = clusterdepth,
    command = {
      cd <- compute_clusterdepth(tr_permutations,
                                 threshold=0.01,
                                 alternative="greater")
      cd$main[,"pvalue"]
      },
    pattern = map(tr_permutations),
    iteration="list",
    description = "cluster depth test" 
  ),
  
  tar_target(
    name = tr_se,
    command = {
      
      # Calculate the mean for each time point (row-wise mean across participants)
      mean_values <- apply(tr_data_con, 1, mean)
      # Calculate the standard deviation for each time point (row-wise standard deviation across participants)
      sd_values <- apply(tr_data_con, 1, sd)
      # Number of participants (columns)
      n_participants <- ncol(tr_data_con)
      # Calculate the standard error of the mean for each time point
      sem_values <- sd_values / sqrt(n_participants)
      
      # Combine the mean and SEM into a data frame for easy reference
      result <- data.frame(times=seq(-0.4, 1.0, length.out = 351), 
                           mean = mean_values, 
                           sem = sem_values)
    },
    pattern = map(tr_data_con),
    iteration="list",
    description = "timeresolved compute standard error of time series" 
  ),
  
  tar_target(
   name = tr_results,
   command = data.frame(times = seq(-0.4, 1.0, length.out = 351),
                        accuracy = tr_permutations[1,] + 0.5,
                        sem = tr_se$sem,
                        ll = tr_permutations[1,] + 0.5 - tr_se$sem,
                        ul = tr_permutations[1,] + 0.5 + tr_se$sem,
                        pmass = clustermass,
                        pdepth = clusterdepth
                 ),
   pattern = map(tr_permutations,clustermass,clusterdepth, tr_se),
   iteration = "list",
   description = "timeresolved results merged depth and mass"
  ),
  
  # maximal y value for plotting
  tar_target(tr_max_y_value, max(sapply(tr_results, function(df) max(df$ul)))),
  tar_target(tr_min_y_value, min(sapply(tr_results, function(df) min(df$ll)))),
  
  # labels for plots
  tar_target(labels, 
             list("Face categorization", "Human face individuation", "Monkey face individuation"),
             iteration = "list"),
  
  # plot
  tar_target(
    name = tr_plots,
    command = {
      offset = 0.5
      
      ggplot(data = tr_results, aes(x = times, y = accuracy)) +
        

        # SE of the mean
        geom_ribbon(aes(ymin = accuracy - tr_se$sem, 
                        ymax = accuracy + tr_se$sem, 
                        #fill = "SEM" # To make it appear in the legend
                        ),
                    fill = "grey",
                    alpha = 0.5) +  # Adjust fill color and transparency
        
        
        geom_line(color = "black") +  # Line plot for times vs accuracy
        
        # Points for pmass < 0.05
        geom_point(aes(y = -0.005 + offset, color = "Cluster: p < 0.05 (FWER)"),
                   data = tr_results[!is.na(tr_results$pmass) & tr_results$pmass < 0.05,],
                   size = 1) +
        
        # Points for pdepth < 0.05
        geom_point(aes(y = -0.010 + offset, color = "Time point: p < 0.05 (FWER)"),
                   data = tr_results[!is.na(tr_results$pdepth) & tr_results$pdepth < 0.05,],
                   size = 1) +
        
        scale_color_manual(values = c("Cluster: p < 0.05 (FWER)" = rkcolors[1], ##7570b3
                                      "Time point: p < 0.05 (FWER)" = rkcolors[3])) +  # #1b9e77
        
        # if i want to appear it in the legend
        #scale_fill_manual(values = c("SEM" = rkcolors[2])) +  # Color for SEM with label
        
        
        #theme_minimal() +
        labs(x = "Time (s)", y = "Accuracy", title = labels) + #, color = ""
        #labs(x = "Time | s", y = "Accuracy", title = "Time-resolved Accuracy (Human vs. Monkey)", color = "Significant Decoding") +
        
        #ylim(min(pdata$accuracy, -0.022), max(pdata$accuracy)) +
        geom_vline(xintercept = 0, linetype="dashed", color="black") +
        geom_hline(yintercept = 0 + offset, linetype="dashed", color="black") +
        
        # xticks
        scale_x_continuous(breaks=c(-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0)) +
        
        theme_light() +
        
        # legend within plot
        # The coordinates for legend.position are x- and y- offsets from the bottom-left of the plot, ranging from 0 - 1.
        theme(strip.text.x = element_blank(),
              #strip.background = element_rect(colour="black", fill="grey"),
              legend.background = element_rect(colour="black", fill="white"),
              legend.position=c(0.12,0.05),
              legend.title = element_blank(), # remove legend title
        ) +
        lims(y = c(min(0.485,tr_min_y_value), tr_max_y_value))
      
    },
    pattern = map(tr_results, labels, tr_se),
    iteration = "list",
    description = "timeresolved results plot"
  ),
  
  tar_target(
    name = tr_plot,
    command = {
      ggarrange(plotlist = tr_plots,
                labels = c("A", "B", "C"),
                ncol = 1, nrow = 3
                #common.legend = TRUE,
                #legend.grob = get_legend(tr_plots[[1]])
                )
                
    },
    description = "merged timeresolved results plots"
  ),
  
  tar_target(
    name = tr_plot_save,
    command = ggsave(tr_plot, filename = "../plots/tr_plot.png", 
                     width = 18, height = 12, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save timeresolved results plot"
  ),
  
  # stimulus image plot, to be merged with tr-plot
  tar_target(
    name = stim_plot_vertical,
    command = {
      ggarrange(plotlist = lapply(stimulus_images,
                                  function(img) grid::rasterGrob(img)),
                ncol = 1, nrow = 3)
    },
    description = "stimulus images plot"
  ),
  tar_target(
    name = stim_plot_horizontal,
    command = {
      ggarrange(plotlist = lapply(stimulus_images,
                                  function(img) grid::rasterGrob(img)),
                ncol = 3, nrow = 1)
    },
    description = "stimulus images plot"
  ),

  # merged time resolved plot and stimulus images
  tar_target(
    name = tr_plot_w_stim,
    command = {
      #tar_skip(TRUE)
      ggarrange(
        tr_plot,
        stim_plot_vertical,
        ncol = 2, nrow = 1, widths = c(0.8, 0.2)  # Set the relative column widths (80% and 20%)
      )
    },
    description = "merged time resolved plot and stimulus images"
  ),
  # save
  tar_target(
    name = tr_plot_w_stim_save,
    command = {
      #tar_skip(TRUE)
      ggsave(tr_plot_w_stim, filename = "../plots/tr_plot_w_stim.png", 
                     width = 18, height = 12, units = "cm", dpi = 300, scale=1.2,
                     bg="white")
      },
    description = "save timeresolved results plot with stimulus"
  ),    
  
  #### EEGNET results
  
  tar_target(
    name = en_file2,
    command = "../models/eegnet.csv",
    format = "file",
    description = "eegnet accuracies raw data file" # requires development targets >= 1.5.0.9001: remotes::install_github("ropensci/targets")
  ),
  
  tar_target(
    name = en_data,
    command = {
      read_csv(en_file2) %>%
        # rename session to participant
        rename(participant = session) %>%
        group_by(context) %>%  # Group by the 'subset' column
        mutate(p_fdr = p.adjust(p_uncorrected, method = "BH")) %>%  # Apply BH correction
        ungroup() %>%  # Ungroup to finish
        mutate(significance = ifelse(p_fdr < 0.05, "p < 0.05 (FDR)", "n.s."))
        
      },
    description = "eegnet accuracies data"
  ),
  
  tar_group_by(
    en_data_con,
    en_data,
    context,
    description = "eegnet accuracies data grouped"
  ),

  # maximal y value for plotting
  tar_target(en_max_y_value, max(max(en_data$accuracy), max(en_data$ul))),
  tar_target(en_min_y_value, min(min(en_data$accuracy), min(en_data$ll))),
  
  # barplots of accuracies across sessions, black for some with p>0.05, darkred for p<0.05
  tar_target(
    name = en_plots,
    command = {
      p <- ggplot(data = en_data_con, 
             aes(x = participant, y = accuracy, color = significance)) + #p_fdr < 0.05
            # Add error bar-like things for the underlying permutation distribution (from ll to ul)
            geom_crossbar(aes(ymin = ll, ymax = ul),
                          color="grey", #rkcolors[2],
                          width = 0.0, #0.2, 0 because this is an additional bar that is added to the lollipop  # Adjust width to change the thickness of the bar
                          size = 1.5) + # Adjust thickness with size, this is the large bar
            
            # lollypop
            geom_point(size = 2) +
            scale_color_manual(values = c("black", rkcolors[3])) +
            geom_segment(aes(x=participant, 
                             xend=participant, 
                             y=0.5, 
                             yend=accuracy)) + 
            
            labs(x = "Participant", y = "Accuracy", title = labels) +
            lims(y = c(en_min_y_value, en_max_y_value)) +
            theme_light() +
            theme(#strip.text.x = element_blank(),
                  legend.background = element_rect(colour="black", fill="white"),
                  legend.position=c(0.6,0.10),
                  legend.title = element_blank(), # remove legend title
                  axis.text.x = element_blank()
        )
      if (labels != "Face categorization"){
        p <- p + guides(color = "none")
      } 
      p
    },
    pattern = map(en_data_con, labels),
    iteration = "list",
    description = "eegnet accuracies plot"
  ),
  
  tar_target(
    name = en_plot,
    command = {
      ggarrange(plotlist = en_plots, 
                labels = c("A", "B", "C"),
                ncol = 3, nrow = 1
                #common.legend = TRUE,
                #legend.grob = get_legend(tr_plots[[1]])
      )
    },
    description = "merged EEGNet results plots"
  ),
  
  tar_target(
    name = en_plot_save,
    command = ggsave(en_plot, filename = "../plots/en_plot.png", 
                     width = 18, height = 10, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save EEGNet results plot"
  ),
  
  # # merged eegnet plot and stimulus images
  # tar_target(
  #   name = en_plot_w_stim,
  #   command = {
  #     #tar_skip(TRUE)
  #     ggarrange(
  #       en_plot,
  #       stim_plot_horizontal,
  #       ncol = 1, nrow = 2, heights = c(0.8, 0.2)  # Set the relative column widths (80% and 20%)
  #     )
  #   },
  #   description = "merged EEGNet plot and stimulus images"
  # ),
  # # save
  # tar_target(
  #   name = en_plot_w_stim_save,
  #   command = {
  #     #tar_skip(TRUE)
  #     ggsave(en_plot_w_stim, filename = "../plots/en_plot_w_stim.png", 
  #                    width = 18, height = 12, units = "cm", dpi = 300, scale=1.2, bg="white")
  #   },
  #   description = "save EEGNet results plot with stimulus"
  # ),    
  
  #### Bayesian population prevalence -- https://github.com/robince/bayesian-prevalence
  
  tar_target(
    name = bayes_prev,
    command = {
      # Bayesian prevalence inference is performed with three numbers: 
      # k, the number of significant participants (e.g. sum of binary indicator
      # variable)
      # n, the number of participants in the sample
      # alpha, the false positive rate
      n <- length(unique(en_data_con$participant)) # n subs
      alpha = 0.05  
      signif <- en_data_con$p_uncorrected 
      indsig = signif < alpha
      k <- sum(indsig)
      
      # plot posterior distribution of population prevalence
      xvals <- seq(0, 1, .01)
      pdf <- bayesprev_posterior(xvals, k, n)
      
      # Add the MAP estimate as a point
      xmap = bayesprev_map(k, n)
      pmap = bayesprev_posterior(xmap, k, n)
      
      # Add the 0.95 HPDI
      int = bayesprev_hpdi(0.95, k, n)
      i1 = int[1]
      i2 = int[2]
      h1 = bayesprev_posterior(i1, k, n)
      h2 = bayesprev_posterior(i2, k, n)
      h <- min(h1, h2) # if the distribution is at the border, then we need to relevel h
      
      data <- data.frame(xvals, pdf)
      # ggplot
      ggplot(data, aes(x=xvals, y=pdf)) +
        #aes_string(y = "pdf") +
        geom_line(size=1.) +
        # 0.95 HDPI
        geom_segment(aes(x=i1, xend=i2, y=h, yend=h), col = rkcolors[1], lwd=1) + ##4e88b9
        annotate("text", x = (i1 + i2) / 2, y = h, 
                 label = paste0("[",round(i1,2),"; ",round(i2,2),"]"), 
                 vjust = 1.6, col = rkcolors[1]) + #4e88b9
        #MAP
        geom_point(aes(x=xmap, y=h), cex = 4, col = rkcolors[3], pch = 16) + ##7c1b00
        annotate("text", x = xmap, y = h, 
                 label = round(xmap,2), 
                 vjust = -1., hjust = +0.5, col = rkcolors[3]) + #7c1b00
        # misc
        labs(x = "Population prevalence", y = "Posterior density", title=labels) +
        
        # in case of low y value, this is necessary to avoid cutting the text of the HDI
        lims(x = c(0, 1), y = c(-2, max(pdf)))
        #geom_hline(yintercept = 0, col = "black", lwd = 0.5)
        
        #theme_minimal()
      
    },
    pattern = map(en_data_con, labels),
    iteration = "list",
    description = "bayesian population prevalence estimation"
  ),
  
  tar_target(
    name = bp_plot,
    command = {
      ggarrange(plotlist = bayes_prev,
                labels = c("A", "B", "C"),
                ncol = 1, nrow = 3
                #common.legend = TRUE,
                #legend.grob = get_legend(tr_plots[[1]])
      )
    },
    description = "merged bayesian prevalence plots"
  ),
  
  tar_target(
    name = pb_plot_save,
    command = ggsave(bp_plot, filename = "../plots/bayesian_prevalence.png", 
                     width = 12, height = 12, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save bayesian prevalence plot"
  ),
  
  ### demographics and trial count analyses
  
  tar_target(
    name = demo_file,
    command = "../data/demographics_trials_inclusions.csv",
    format = "file",
    description = "demographics trial counts and inclusion info file" 
  ),
  tar_target(
    name = demo_data,
    command = {
      read.csv(demo_file) %>%
        filter(incl_session=="True") %>%
        select(-c(incl_session, incl_subject, invited, included, sub_ses_str, ses)) %>%
        # group by sub, sum up trials, average age, take first of sex
        group_by(sub) %>%
        summarise(trials = sum(trials), 
                  age = mean(age),
                  sex = first(sex)) %>%
        ungroup() %>%
        mutate(sub = sprintf("sub-%03d", sub))
    },
    description = "demographics trial counts and inclusion info data" 
  ),
  
  # save into csv
  tar_target(
    name = demo_data_save,
    command = write.csv(demo_data, "../data/demo_trials_concise.csv", row.names = FALSE),
    description = "save demographics trial counts and inclusion info data"
  ),
  
  # correlate accuracy with n_trials
  tar_target(
    name = corr_trials,
    command = {
      data <- en_data_con %>%
        left_join(demo_data, by = c("participant" = "sub")) %>%
        select(accuracy, trials)
      
      cor.test(data$accuracy, data$trials, method="spearman")
    },
    pattern = map(en_data_con),
    iteration = "list",
    description = "correlation between EEGNet accuracy and number of trials"
  ),
  
  # properties of the trials
  tar_target(
    name = trial_props,
    command = {
      mean <- mean(demo_data$trials)
      median <- median(demo_data$trials)
      sd <- sd(demo_data$trials)
      min <- min(demo_data$trials)
      max <- max(demo_data$trials)
      print(paste("mean:", mean, "median:", median, "sd:", sd, "min:", min, "max:", max))
    }
  ),
  
  ### accuracies per subclass in the EEGNet - inter contrast
  
  tar_target(
    name = acc_subclass_file,
    command = "../models/eegnet_subclass_acc.csv",
    format = "file",
    description = "EEGNet subclass accuracies file"
  ),
  
  tar_target(
    name = acc_subclass_data,
    command = {
      read.csv(acc_subclass_file) %>%
        # replace h1, h2, m1, m2 in col class with human1, human2, monkey1, monkey2
        mutate(class = gsub("h1", "human1", class),
               class = gsub("h2", "human2", class),
               class = gsub("m1", "monkey1", class),
               class = gsub("m2", "monkey2", class))
    },
    description = "EEGNet subclass accuracies data"
  ),
  
  tar_target(
    name = acc_subclass_stats,
    command = {
      pairwise_results <- pairwise.t.test(acc_subclass_data$accuracy,
                                          g=acc_subclass_data$class,
                                          paired = TRUE, 
                                          alternative = "two.sided",
                                          p.adjust.method = "BH")
        
    },
    description = "EEGNet subclass accuracies statistics"
  ),
  
  tar_target(
    name = acc_subclass_plot,
    command = {
      ggplot(acc_subclass_data, 
             aes(x = class, y = accuracy)) +
        #geom_bar(stat="identity", fill = rkcolors[1]) +
        geom_boxplot(notch = TRUE, fill="lightgrey") + # fill = rkcolors[1], 
        
        # Add points for each accuracy value
        #geom_point(aes(color = session), size = 3, alpha = 0.6) +
        # Add lines connecting paired data points for each session
        geom_line(aes(group = session, color = session), alpha = 0.6) +
        
        labs(x = "Subclass", y = "Accuracy") + #, title = "EEGNet subclass accuracies") +
        theme_minimal() +
        geom_hline(yintercept = 0.5, linetype="dashed", color = "black", size=1.5) +
        
        # remove legend, must be after all "geom" layers
        theme(legend.position = "none")
        
    },
    description = "EEGNet subclass accuracies plot"
  ),
  
  tar_target(
    name = acc_subclass_save,
    command = ggsave(acc_subclass_plot, filename = "../plots/eegnet_subclass_acc.png", 
                     width = 12, height = 12, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save EEGNet subclass accuracies plot"
  ),
  
  # R1
  
  tar_target(
    name = tr_file_single,
    command = "../models/timeresolved_single.csv",
    format = "file",
    description = "timeresolved timeseries data file (only subs with 2 sess)" 
  ),
  
  # tr
  tar_target(
    name = tr_data_single,
    command = read.csv(tr_file_single),
    description = "timeresolved timeseries data single" 
  ),
  
  # manual group by
  tar_target(name = tr_data_inter_single, command = tr_data_single %>% subset(species=="inter") %>% select(-c(species))),
  tar_target(name = tr_data_intra_human_single, command = tr_data_single %>% subset(species=="intra_human") %>% select(-c(species))),
  tar_target(name = tr_data_intra_monkey_single, command = tr_data_single %>% subset(species=="intra_monkey") %>% select(-c(species))),
  tar_target(name = tr_data_con_single, 
             command = list(tr_data_inter_single, tr_data_intra_human_single, tr_data_intra_monkey_single),
             iteration = "list"),
  
  # Split each species-specific dataset by session
  tar_target(
    name = tr_data_con_split_pre,
    command = {
      split_sessions(tr_data_con_single) #, unlist(xx, recursive=FALSE)
      },
    pattern = map(tr_data_con_single),
    iteration = "list"
  ),
  
  # split the 2 lists per branch into one large list
  tar_target(
    name = tr_data_con_split,
    command = {
      # unlist(tr_data_con_split_pre, recursive=FALSE)
      tr_data_con_split_pre %>% purrr::reduce(c)
    },
    iteration = "list"
  ),
  
  
  tar_target(
    name = tr_permutations_split,
    command = sign_flip(tr_data_con_split),
    pattern = map(tr_data_con_split),
    iteration = "list"
  ),


  tar_target(
    name = clustermass_split,
    command = {
      cm <- compute_clustermass(tr_permutations_split,
                                threshold=0.01,
                                aggr_FUN=sum,
                                alternative = "greater")
      cm$main[,"pvalue"]
    },
    pattern = map(tr_permutations_split),
    iteration="list",
    description = "cluster mass test"
  ),
  tar_target(
    name = clusterdepth_split,
    command = {
      cd <- compute_clusterdepth(tr_permutations_split,
                                 threshold=0.01,
                                 alternative="greater")
      cd$main[,"pvalue"]
    },
    pattern = map(tr_permutations_split),
    iteration="list",
    description = "cluster depth test"
  ),
  
  tar_target(
    name = tr_se_split,
    command = {

      # Calculate the mean for each time point (row-wise mean across participants)
      mean_values <- apply(tr_data_con_split, 1, mean)
      # Calculate the standard deviation for each time point (row-wise standard deviation across participants)
      sd_values <- apply(tr_data_con_split, 1, sd)
      # Number of participants (columns)
      n_participants <- ncol(tr_data_con_split)
      # Calculate the standard error of the mean for each time point
      sem_values <- sd_values / sqrt(n_participants)

      # Combine the mean and SEM into a data frame for easy reference
      result <- data.frame(times=seq(-0.4, 1.0, length.out = 351),
                           mean = mean_values,
                           sem = sem_values)
    },
    pattern = map(tr_data_con_split),
    iteration="list",
    description = "timeresolved compute standard error of time series"
  ),

  tar_target(
    name = tr_results_split,
    command = data.frame(times = seq(-0.4, 1.0, length.out = 351),
                         accuracy = tr_permutations_split[1,] + 0.5,
                         sem = tr_se_split$sem,
                         ll = tr_permutations_split[1,] + 0.5 - tr_se_split$sem,
                         ul = tr_permutations_split[1,] + 0.5 + tr_se_split$sem,
                         pmass = clustermass_split,
                         pdepth = clusterdepth_split
    ),
    pattern = map(tr_permutations_split,clustermass_split,clusterdepth_split, tr_se_split),
    iteration = "list",
    description = "timeresolved results merged depth and mass"
  ),

  # maximal y value for plotting
  tar_target(tr_max_y_value_split, max(sapply(tr_results_split, function(df) max(df$ul)))),
  tar_target(tr_min_y_value_split, min(sapply(tr_results_split, function(df) min(df$ll)))),

  # labels for plots
  tar_target(labels_split,
            list("Face categorization: 5-8 months","Face categorization: 9-11 months",
                 "Human face individuation: 5-8 months","Human face individuation: 9-11 months",
                 "Monkey face individuation: 5-8 months","Monkey face individuation: 9-11 months"),
            iteration = "list"),

  # plot
  tar_target(
    name = tr_plots_split,
    command = {
      offset = 0.5

      ggplot(data = tr_results_split, aes(x = times, y = accuracy)) +


        # SE of the mean
        geom_ribbon(aes(ymin = accuracy - tr_se_split$sem,
                        ymax = accuracy + tr_se_split$sem,
                        #fill = "SEM" # To make it appear in the legend
        ),
        fill = "grey",
        alpha = 0.5) +  # Adjust fill color and transparency


        geom_line(color = "black") +  # Line plot for times vs accuracy

        # Points for pmass < 0.05
        geom_point(aes(y = -0.005 + offset, color = "Cluster: p < 0.05 (FWER)"),
                   data = tr_results_split[!is.na(tr_results_split$pmass) & tr_results_split$pmass < 0.05,],
                   size = 1) +

        # Points for pdepth < 0.05
        geom_point(aes(y = -0.010 + offset, color = "Time point: p < 0.05 (FWER)"),
                   data = tr_results_split[!is.na(tr_results_split$pdepth) & tr_results_split$pdepth < 0.05,],
                   size = 1) +

        scale_color_manual(values = c("Cluster: p < 0.05 (FWER)" = rkcolors[1], ##7570b3
                                      "Time point: p < 0.05 (FWER)" = rkcolors[3])) +  # #1b9e77

        # if i want to appear it in the legend
        #scale_fill_manual(values = c("SEM" = rkcolors[2])) +  # Color for SEM with label


        #theme_minimal() +
        labs(x = "Time (s)", y = "Accuracy", title = labels_split) + #, color = ""
        #labs(x = "Time | s", y = "Accuracy", title = "Time-resolved Accuracy (Human vs. Monkey)", color = "Significant Decoding") +

        #ylim(min(pdata$accuracy, -0.022), max(pdata$accuracy)) +
        geom_vline(xintercept = 0, linetype="dashed", color="black") +
        geom_hline(yintercept = 0 + offset, linetype="dashed", color="black") +

        # xticks
        scale_x_continuous(breaks=c(-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0)) +

        theme_light() +

        # legend within plot
        # The coordinates for legend.position are x- and y- offsets from the bottom-left of the plot, ranging from 0 - 1.
        theme(strip.text.x = element_blank(),
              #strip.background = element_rect(colour="black", fill="grey"),
              legend.background = element_rect(colour="black", fill="white"),
              legend.position=c(0.12,0.05),
              legend.title = element_blank(), # remove legend title
        ) +
        lims(y = c(min(0.485,tr_min_y_value_split), tr_max_y_value_split))

    },
    pattern = map(tr_results_split, labels_split, tr_se_split),
    iteration = "list",
    description = "timeresolved results plot"
  ),

  tar_target(
    name = tr_plot_split,
    command = {
      ggarrange(plotlist = tr_plots_split,
                labels = c("A", "B", "C", "D", "E", "F"),
                ncol = 2, nrow = 3
                #common.legend = TRUE,
                #legend.grob = get_legend(tr_plots[[1]])
      )

    },
    description = "merged timeresolved results plots"
  ),

  tar_target(
    name = tr_plot_save_split,
    command = ggsave(tr_plot_split, filename = "../plots/tr_plot_split.png", 
                     width = 18, height = 20, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save timeresolved results plot"
  ),
  
  
  #### EEGNET results
  
  tar_target(
    name = en_file_split,
    command = "../models/eegnet_single.csv",
    format = "file",
    description = "eegnet accuracies raw data file" # requires development targets >= 1.5.0.9001: remotes::install_github("ropensci/targets")
  ),
  
  tar_target(
    name = en_data_split,
    command = {
      read_csv(en_file_split) %>%
        # rename R1: new order because now also split
        mutate(participant = str_extract(session, "sub-\\d+")) %>%
        mutate(session = str_extract(session, "ses-\\d+")) %>%
        group_by(context, session) %>%  # Group by the 'subset' column
        mutate(p_fdr = p.adjust(p_uncorrected, method = "BH")) %>%  # Apply BH correction 
        ungroup() %>%  # Ungroup to finish
        mutate(significance = ifelse(p_fdr < 0.05, "p < 0.05 (FDR)", "n.s."))
      
    },
    description = "eegnet accuracies data"
  ),

  # ALTERNATIVE, like in TR
  # # manual group by
  # tar_target(name = tr_data_inter_single, command = tr_data_single %>% subset(species=="inter") %>% select(-c(species))),
  # tar_target(name = tr_data_intra_human_single, command = tr_data_single %>% subset(species=="intra_human") %>% select(-c(species))),
  # tar_target(name = tr_data_intra_monkey_single, command = tr_data_single %>% subset(species=="intra_monkey") %>% select(-c(species))),
  # tar_target(name = tr_data_con_single, 
  #            command = list(tr_data_inter_single, tr_data_intra_human_single, tr_data_intra_monkey_single),
  #            iteration = "list"),
  # 
  # # Split each species-specific dataset by session
  # tar_target(
  #   name = tr_data_con_split_pre,
  #   command = {
  #     split_sessions(tr_data_con_single) #, unlist(xx, recursive=FALSE)
  #   },
  #   pattern = map(tr_data_con_single),
  #   iteration = "list"
  # ),
  # 
  # # split the 2 lists per branch into one large list
  # tar_target(
  #   name = tr_data_con_split,
  #   command = {
  #     # unlist(tr_data_con_split_pre, recursive=FALSE)
  #     tr_data_con_split_pre %>% purrr::reduce(c)
  #   },
  #   iteration = "list"
  # ),  
  
  
  tar_group_by(
    en_data_con_split,
    en_data_split,
    context,
    session, # R1: another split by session
    description = "eegnet accuracies data grouped"
  ),
  
  # maximal y value for plotting
  tar_target(en_max_y_value_split, max(max(en_data_split$accuracy), max(en_data_split$ul))),
  tar_target(en_min_y_value_split, min(min(en_data_split$accuracy), min(en_data_split$ll))),
  
  # barplots of accuracies across sessions, black for some with p>0.05, darkred for p<0.05
  tar_target(
    name = en_plots_split,
    command = {
      p <- ggplot(data = en_data_con_split,
                  aes(x = participant, y = accuracy, color = significance)) + 
        geom_crossbar(aes(ymin = ll, ymax = ul),
                      color="grey", #rkcolors[2],
                      width = 0.0, #0.2, 0 because this is an additional bar that is added to the lollipop  # Adjust width to change the thickness of the bar
                      size = 1.5) + # Adjust thickness with size, this is the large bar
        # lollypop
        geom_point(size = 2) +
        scale_color_manual(values = c("black", rkcolors[3])) +
        geom_segment(aes(x=participant,
                         xend=participant,
                         y=0.5,
                         yend=accuracy)) +
  
        labs(x = "Participant", y = "Accuracy", title = labels_split) +
        lims(y = c(en_min_y_value_split, en_max_y_value_split)) +
  
        theme_light() +
  
        theme(#strip.text.x = element_blank(),
          legend.background = element_rect(colour="black", fill="white"),
          #strip.background = element_rect(colour="white", fill="white"),
          legend.position=c(0.6,0.10),
          legend.title = element_blank(), # remove legend title
          axis.text.x = element_blank()
          #element_text(angle = 90, size=8, vjust = 0)
        )
      if (labels_split != "Face categorization"){
        p <- p + guides(color = "none")
      }
      p
    },
    pattern = map(en_data_con_split, labels_split),
    iteration = "list",
    description = "eegnet accuracies plot"
  ),
  
  tar_target(
    name = en_plot_split,
    command = {
      ggarrange(plotlist = en_plots_split,
                labels = c("A", "B", "C", "D", "E", "F"),
                ncol = 2, nrow = 3 # R1
                #common.legend = TRUE,
                #legend.grob = get_legend(tr_plots[[1]])
      )
    },
    description = "merged EEGNet results plots"
  ),
  
  tar_target(
    name = en_plot_save_split,
    command = ggsave(en_plot_split, filename = "../plots/en_plot_split.png",
                     width = 18, height = 16, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save EEGNet results plot"
  ),
  

  tar_target(
    name = en_stats_split,
    command = {
      en_data_split %>%
        select(participant, context, session, accuracy) %>%
        pivot_wider(
          names_from = session,
          values_from = accuracy
        ) %>%
        group_by(context) %>%
        # paired t-test
        summarise(
          t_test = list(t.test(`ses-001`, `ses-002`, paired = TRUE)),
        ) %>%
        mutate(
          tidy_result = lapply(t_test, broom::tidy)
        ) %>%
          unnest(tidy_result) %>%
          mutate(
            apa_report = sprintf("t(%d) = %.2f, p = %.3f", parameter, statistic, p.value)
          ) %>%
          select(context, apa_report)
    },
    description = "split sessions, paired ttest ses 1 vs 2"
  ),
  
  tar_target(
    name = en_plot_corr_split,
    command = {
      en_data_split %>%
        select(participant, context, session, accuracy) %>%
        pivot_wider(
          names_from = session,
          values_from = accuracy
        ) %>%
        ggplot(aes(x = `ses-001`, y = `ses-002`)) +
        geom_point(size = 2) +
        facet_grid(~context, labeller = as_labeller(c(
          inter = "Face categorization",
          intra_human = "Human face individuation",
          intra_monkey = "Monkey face individuation"
        ))) +
        # x=y line
        geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed") +
        # square facets
        coord_fixed() +
        labs(
          x = "Accuracy: 5-8 months",
          y = "Accuracy: 9-11 months"
        ) +
        # Add APA report text
        geom_text(
          data = en_stats_split,
          aes(x = 0.4, y = 0.8, label = apa_report),  # Adjust x/y to fit your plot range
          inherit.aes = FALSE,
          size = 3.5,
          hjust = 0
        ) +
        theme_bw()
    },
    description = "split sessions, correlation ses 1 vs 2"
  ),
  
  tar_target(
    name = en_plot_corr_save_split,
    command = ggsave(en_plot_corr_split, filename = "../plots/en_plot_scatter_split.png", 
                     width = 18, height = 10, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save eegnet scatter results plot"
  ),
  
  ####################
  # R1 adult pilots
  ####################
  
  tar_target(
    name = tr_file_adults,
    command = "../models/timeresolved_adults.csv",
    format = "file",
    description = "timeresolved timeseries data file" 
  ),
  
  # tr
  tar_target(
    name = tr_data_adults,
    command = {read.csv(tr_file_adults) %>%
        rename(participant = session) %>%
        rename(context = species) %>%
        select(-c(subset, surrogate))
      },
    description = "timeresolved timeseries data" 
  ),
  
  tar_group_by(
    tr_data_adults_grouped,
    tr_data_adults,
    context,
    participant, # R1: another split by session
    description = "timeresolved accuracies data grouped"
  ),
  # maximal y value for plotting
  tar_target(tr_max_y_value_adults, max(tr_data_adults$accuracy)),
  tar_target(tr_min_y_value_adults, min(tr_data_adults$accuracy)),
  # labels for plots
  tar_target(labels_adults,
             list("Face categorization: adult 1","Face categorization: adult 2",
                  "Human face individuation: adult 1","Human face individuation: adult 2",
                  "Monkey face individuation: adult 1","Monkey face individuation: adult 2"),
             iteration = "list"),  
  
  # plot
  tar_target(
    name = tr_plots_adults,
    command = {
      offset = 0.5
      
      p <- ggplot(data = tr_data_adults_grouped, aes(x = times, y = accuracy)) +
        
        geom_line(color = "black") +  # Line plot for times vs accuracy
        
        # Points for pmass < 0.05
        geom_point(aes(y = -0.02 + offset, color = "Cluster: p < 0.05 (FWER)"),
                   #data = tr_data_adults_grouped[significant=="True"],
                   data = dplyr::filter(tr_data_adults_grouped, significant == "True"),
                   size = 1) +

        scale_color_manual(values = c("Cluster: p < 0.05 (FWER)" = rkcolors[1], ##7570b3
                                      "Time point: p < 0.05 (FWER)" = rkcolors[3])
                                      ) +  # #1b9e77
        
        #theme_minimal() +
        labs(x = "Time (s)", y = "Accuracy", title = labels_adults) + #, color = ""
        #labs(x = "Time | s", y = "Accuracy", title = "Time-resolved Accuracy (Human vs. Monkey)", color = "Significant Decoding") +
        
        geom_vline(xintercept = 0, linetype="dashed", color="black") +
        geom_hline(yintercept = 0 + offset, linetype="dashed", color="black") +
        
        # xticks
        scale_x_continuous(breaks=c(-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0)) +
        
        theme_light() +
        
        # legend within plot
        # The coordinates for legend.position are x- and y- offsets from the bottom-left of the plot, ranging from 0 - 1.
        theme(strip.text.x = element_blank(),
              #strip.background = element_rect(colour="black", fill="grey"),
              legend.background = element_rect(colour="black", fill="white"),
              legend.position=c(0.15,0.1),
              legend.title = element_blank(), # remove legend title
        ) +
        lims(y = c(min(0.485,tr_min_y_value_adults), tr_max_y_value_adults))
      
      if (labels_adults != "Face categorization: adult 2"){
        p <- p + guides(color = "none")
      } 
      p
      
    },
    pattern = map(tr_data_adults_grouped, labels_adults),
    iteration = "list",
    description = "timeresolved results plot"
  ),
  
  tar_target(
    name = tr_plot_adults,
    command = {
      ggarrange(plotlist = tr_plots_adults,
                labels = c("A", "B", "C", "D", "E", "F"),
                ncol = 2, nrow = 3
                #common.legend = TRUE,
                #legend.grob = get_legend(tr_plots[[1]])
      )
    },
    description = "merged timeresolved results plots"
  ),
  
  tar_target(
    name = tr_plot_save_adults,
    command = ggsave(tr_plot_adults, filename = "../plots/tr_plot_adults.png", 
                     width = 18, height = 20, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save timeresolved results plot"
  ),
  
  # EN
  
  
  
  tar_target(
    name = en_file_adults,
    command = "../models/eegnet_adults.csv",
    format = "file",
    description = "eegnet accuracies raw data file" # requires development targets >= 1.5.0.9001: remotes::install_github("ropensci/targets")
  ),
  
  tar_target(
    name = en_data_adults,
    command = {
      read_csv(en_file_adults) %>%
        # rename session to participant
        rename(participant = session) %>%
        group_by(context) %>%  # Group by the 'subset' column
        mutate(p_fdr = p.adjust(p_uncorrected, method = "BH")) %>%  # Apply BH correction
        ungroup() %>%  # Ungroup to finish
        mutate(significance = ifelse(p_fdr < 0.05, "p < 0.05 (FDR)", "n.s."))
      
    },
    description = "eegnet accuracies data"
  ),
  
  tar_group_by(
    en_data_con_adults,
    en_data_adults,
    context,
    description = "eegnet accuracies data grouped"
  ),
  
  # maximal y value for plotting
  tar_target(en_max_y_value_adults, max(max(en_data_adults$accuracy), max(en_data_adults$ul))),
  tar_target(en_min_y_value_adults, min(min(en_data_adults$accuracy), min(en_data_adults$ll))),
  
  # barplots of accuracies across sessions, black for some with p>0.05, darkred for p<0.05
  tar_target(
    name = en_plots_adults,
    command = {
      p <- ggplot(data = en_data_con_adults, 
                  aes(x = participant, y = accuracy, color = significance)) + #p_fdr < 0.05
        #geom_bar(stat = "identity") +
        #scale_fill_manual(values = c("black", "darkred")) +
        
        # Add error bar-like things for the underlying permutation distribution (from ll to ul)
        #geom_errorbar(aes(ymin = ll, ymax = ul), width = 0.5, color = rkcolors[2]) +
        geom_crossbar(aes(ymin = ll, ymax = ul),
                      color="grey", #rkcolors[2],
                      width = 0.0, #0.2, 0 because this is an additional bar that is added to the lollipop  # Adjust width to change the thickness of the bar
                      size = 1.5) + # Adjust thickness with size, this is the large bar
        
        
        
        # lollypop
        geom_point(size = 2) +
        #scale_color_manual(values = c("black", rkcolors[3])) +
        scale_color_manual(values = c(
          "n.s." = "black",
          "p < 0.05 (FDR)" = rkcolors[3]
          )) + # R1 set value explicitly
        geom_segment(aes(x=participant, 
                         xend=participant, 
                         y=0.5, 
                         yend=accuracy)) + 
        
        
        labs(x = "Participant", y = "Accuracy", title = labels) +
        lims(y = c(en_min_y_value_adults, en_max_y_value_adults)) +
        
        
        theme_light() +
        
        theme(#strip.text.x = element_blank(),
          legend.background = element_rect(colour="black", fill="white"),
          #strip.background = element_rect(colour="white", fill="white"),
          legend.position=c(0.5,0.80),
          legend.title = element_blank(), # remove legend title
          axis.text.x = element_blank()
          #element_text(angle = 90, size=8, vjust = 0)
        )
      if (labels != "Monkey face individuation"){
        p <- p + guides(color = "none")
      } 
      p
    },
    pattern = map(en_data_con_adults, labels),
    iteration = "list",
    description = "eegnet accuracies plot"
  ),
  
  tar_target(
    name = en_plot_adults,
    command = {
      ggarrange(plotlist = en_plots_adults, 
                labels = c("A", "B", "C"),
                ncol = 3, nrow = 1
                #common.legend = TRUE,
                #legend.grob = get_legend(tr_plots[[1]])
      )
    },
    description = "merged EEGNet results plots"
  ),
  
  tar_target(
    name = en_plot_save_adults,
    command = ggsave(en_plot_adults, filename = "../plots/en_plot_adults.png", 
                     width = 18, height = 8, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save EEGNet results plot"
  ),

  
  
  
  
  # R1 sort EN participants by age in plot

  tar_target(en_data_con_sorted, # sort participants by average age across sessions
             {
               # sort participants by age
               sorted_by_age <- demo_data %>%
                 arrange(age) %>%
                 pull(sub)
               
               en_data_con %>%
                 left_join(demo_data %>% rename(participant = sub), 
                           by = "participant") %>% # sort participants factor by age
                 mutate(participant = factor(participant, levels = sorted_by_age)) %>%
                 arrange(participant) %>%
                 mutate(participant_age = paste0(participant, " (", age, " mo.)"))
                 
             },
             pattern = map(en_data_con),
             iteration = "list",
             description = "sort EN participants by age"
  ),
  
  tar_target(
    name = en_plots_sorted,
    command = {
      p <- ggplot(data = en_data_con_sorted, 
                  aes(x = participant, y = accuracy, color = significance)) + # R1
        # Add error bar-like things for the underlying permutation distribution (from ll to ul)
        geom_crossbar(aes(ymin = ll, ymax = ul),
                      color="grey", #rkcolors[2],
                      width = 0.0, #0.2, 0 because this is an additional bar that is added to the lollipop  # Adjust width to change the thickness of the bar
                      size = 1.5) + # Adjust thickness with size, this is the large bar
        
        # lollypop
        geom_point(size = 2) +
        scale_color_manual(values = c("black", rkcolors[3])) +
        geom_segment(aes(x=participant, 
                         xend=participant, 
                         y=0.5, 
                         yend=accuracy)) + 
        # R1: add age
        #scale_x_discrete(labels = levels(en_data_con_sorted$participant)) +
        
        labs(x = "", y = "Accuracy", title = labels) +
        lims(y = c(en_min_y_value, en_max_y_value)) +
        theme_light() +
        theme(#strip.text.x = element_blank(),
          legend.background = element_rect(colour="black", fill="white"),
          legend.position=c(0.6,0.10),
          legend.title = element_blank(), # remove legend title
          axis.text.x = element_blank() # R1
          #axis.text.x = element_text(angle = 270, vjust = 0, hjust = 0)
        )
      if (labels != "Face categorization"){
        p <- p + guides(color = "none")
      } 
      p
    },
    pattern = map(en_data_con_sorted, labels),
    iteration = "list",
    description = "eegnet accuracies plot"
  ),
  
  # additional age lineplots
  tar_target(
    name = en_plots_sorted_age,
    command = {
      p <- ggplot(data = en_data_con_sorted, 
                  aes(x = participant, y = age, group = 1)) + # R1
        geom_line(size = 1) +
        labs(x = "Participant", y = "Age (months)") + #, title = labels
        #lims(y = c(en_min_y_value, en_max_y_value)) +
        theme_light() +
        theme(strip.text.x = element_blank(),
        #  legend.background = element_rect(colour="black", fill="white"),
        #  legend.position=c(0.6,0.10),
        #  legend.title = element_blank(), # remove legend title
          axis.text.x = element_blank() # R1
        #  axis.text.x = element_text(angle = 270, vjust = 0, hjust = 0)
        )
      p
    },
    pattern = map(en_data_con_sorted),
    iteration = "list",
    description = "eegnet accuracies plot"
  ),

  tar_target(
    name = en_plot_sorted,
    command = {
      ggarrange(plotlist = en_plots_sorted, 
                labels = c("A", "B", "C"),
                ncol = 3, nrow = 1
      )
    },
    description = "merged EEGNet results plots"
  ),
  
  tar_target(
    name = en_plot_sorted_age,
    command = {
      ggarrange(plotlist = en_plots_sorted_age, 
                #labels = c("", "", ""),
                ncol = 3, nrow = 1
      )
    },
    description = "merged EEGNet results age plots"
  ),
  
  tar_target(name = en_plot_sorted_merge,
             command = {
               ggarrange(plotlist = list(en_plot_sorted, en_plot_sorted_age), 
                         ncol = 1, nrow = 2,
                         heights = c(0.8, 0.2)  # First plot takes 80%, second plot takes 20%
               )
    },             
             
  ),
  
  tar_target(
    name = en_plot_save_sorted,
    command = ggsave(en_plot_sorted_merge, filename = "../plots/en_plot_sorted.png", 
                     width = 18, height = 10, units = "cm", dpi = 300, scale=1.2,
                     bg="white"),
    description = "save EEGNet results plot"
  )



  
  

  
)
