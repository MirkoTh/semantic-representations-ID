rm(list = ls())

library(tidyverse)
library(data.table)
library(grid)
library(gridExtra)

source("R/utils.R")

rebase_subject_ids <- function(my_tbl) {
  my_tbl$X4 <- factor(
    my_tbl$X4, 
    labels = seq(0, (length(unique(my_tbl$X4)) - 1))
  )
  my_tbl$X4 <- as.numeric(my_tbl$X4) - 1
  return(my_tbl)
}


# Load Data ---------------------------------------------------------------

tbl_train_things <- read_delim("data/train_90_ID_item.txt", col_names = FALSE)
tbl_eval_things <- read_delim("data/test_10_ID_item.txt", col_names = FALSE)

# to check
tbl_train_things %>% count(X4) %>% left_join(
  tbl_eval_things %>% count(X4),
  by = "X4", suffix = c("_train", "_eval")
) %>% mutate(n_tot = n_train + n_eval) %>%
  arrange(n_tot)

tbl_ooo_study1 <- read_delim(
  "data/study1-2025-08/ooo_data_modeling_excluded.txt",
  col_names = FALSE
) %>% arrange(X4)
participant_ids_new <- unique(tbl_ooo_study1$X4)
tbl_ooo_study1 <- tbl_ooo_study1 %>% rebase_subject_ids() %>%
  mutate(X4 = X4 + 1)


max_id_things_train <- max(tbl_train_things$X4)
max_id_things_eval <- max(tbl_eval_things$X4)
if (!assertthat::are_equal(max_id_things_train,  max_id_things_eval)) stop()


# Merge and Save Full Data Set --------------------------------------------

tbl_ooo_study1$X4 <- tbl_ooo_study1$X4 + max_id_things_train
# full data set
tbl_full <- rbind(tbl_train_things, tbl_eval_things, tbl_ooo_study1)


write_delim(
  tbl_full, 
  file = "data/study1-2025-08/ooo_data_modeling_old_and_new.txt", 
  col_names = FALSE
)

pids_study1 <- tibble(
  # ids used in modeling
  participant_id_model = unique(tbl_ooo_study1$X4),
  # ids only used for new study participants
  participant_id_new = participant_ids_new
)

# save the updated participant ids to easily extract data after model fitting
write_csv(pids_study1, "data/study1-2025-08/new-participant-ids-in-joint-modeling.csv")



# Save Data Frame for Testcase --------------------------------------------


# small data set for testing
# make sure, new ids are in testcase set (for dev purposes)
n_samples_each <- 10
set.seed(10)
tbl_testcase <- tbl_full[tbl_full$X4 %in% sample(max_id_things_train, n_samples_each), ]
tbl_ooo_study1_testcase <- tbl_ooo_study1[tbl_ooo_study1$X4 %in% sample((max_id_things_train+1):max(tbl_ooo_study1$X4), n_samples_each), ]
tbl_testcase <- rbind(tbl_testcase, tbl_ooo_study1_testcase)


tbl_testcase <- rebase_subject_ids(tbl_testcase)

write_delim(
  tbl_testcase, 
  file = "data/study1-2025-08/ooo_data_modeling_old_and_new_testcase.txt", 
  col_names = FALSE
)


# only ids from new data set
unique_test_pids <- unique(tbl_testcase$X4)
participant_ids_new_testcase <- tbl_ooo_study1_testcase %>% left_join(pids_study1, by = c("X4" = "participant_id_model")) %>%
  group_by(participant_id_new) %>% count() %>% select(participant_id_new) %>% as_vector()
pids_study1_testcase <- tibble(
  # ids used in modeling
  participant_id_model = unique_test_pids[which(unique_test_pids >= n_samples_each)],
  # ids only used for new study participants
  participant_id_new = participant_ids_new_testcase
)

# save the updated participant ids to easily extract data after model fitting
write_csv(
  pids_study1_testcase, 
  "data/study1-2025-08/new-participant-ids-in-joint-modeling-testcase.csv"
)


# Create and Save Half-Split Data Set ------------------------------------


# do this twice with two different seed values
# for the split-half reliability analysis of the weight change we need to run
# each half twice to have the weight change measure twice


# number 1
set.seed(10)
tbl_reorder <- tbl_full %>%
  mutate(idx = 1:nrow(.)) %>%
  group_by(X4) %>%
  mutate(
    trial_id = row_number(idx),
    trial_id_random = sample(max(trial_id), replace=FALSE),
    first_half = trial_id_random <= max(trial_id) / 2
  ) %>% ungroup() %>%
  select(-c(idx, trial_id)) %>%
  rename(trial_id = trial_id_random) %>%
  arrange(X4, trial_id)

l_reordered <- tbl_reorder %>% split(.$first_half)
tbl_first_half <- l_reordered[[1]] %>% select(-c(trial_id, first_half))
tbl_second_half <- l_reordered[[2]] %>% select(-c(trial_id, first_half))


# save the updated participant ids to easily extract data after model fitting
write_delim(
  tbl_first_half, 
  "data/study1-2025-08/ooo_data_modeling_old_and_new_h1.txt",
  col_names = FALSE
)
write_delim(
  tbl_second_half, 
  "data/study1-2025-08/ooo_data_modeling_old_and_new_h2.txt",
  col_names = FALSE
)

# number 2
set.seed(597)
tbl_reorder <- tbl_full %>%
  mutate(idx = 1:nrow(.)) %>%
  group_by(X4) %>%
  mutate(
    trial_id = row_number(idx),
    trial_id_random = sample(max(trial_id), replace=FALSE),
    first_half = trial_id_random <= max(trial_id) / 2
  ) %>% ungroup() %>%
  select(-c(idx, trial_id)) %>%
  rename(trial_id = trial_id_random) %>%
  arrange(X4, trial_id)

l_reordered <- tbl_reorder %>% split(.$first_half)
tbl_first_half <- l_reordered[[1]] %>% select(-c(trial_id, first_half))
tbl_second_half <- l_reordered[[2]] %>% select(-c(trial_id, first_half))


# save the updated participant ids to easily extract data after model fitting
write_delim(
  tbl_first_half, 
  "data/study1-2025-08/ooo_data_modeling_old_and_new_h1_v2.txt",
  col_names = FALSE
)
write_delim(
  tbl_second_half, 
  "data/study1-2025-08/ooo_data_modeling_old_and_new_h2_v2.txt",
  col_names = FALSE
)



# Comparison of two samples -----------------------------------------------

# test whether our new sample differs systematically from Hebart et al. (2020) sample

tbl_full <- tbl_full %>%
  mutate(
    dataset = ifelse(X4 <= max_id_things_train, "Old", "New")
  )


full_dt <- data.table(tbl_full)

# Compute min, max, and sum in vectorized form
full_dt[, `:=`(
  id_lo  = pmin(X1, X2, X3),
  id_hi  = pmax(X1, X2, X3),
  id_sum = X1 + X2 + X3
)]

# Middle value = sum - min - max
full_dt[, id_mid := id_sum - id_lo - id_hi]

# Drop helper column
full_dt[, id_sum := NULL]

full_dt[, triplet_id := as.numeric(as.factor(str_c(as.character(id_lo), "-", as.character(id_mid), "-", as.character(id_hi))))]

full_dt[, n := .N, by = triplet_id]

set.seed(1234)
full_dt[, half := sample(.N, replace = FALSE) %% 2 + 1, by = .(dataset, triplet_id)]

# if participants have observed the same triplet several times, just use the first encounter
full_dt$idx <- 1:nrow(full_dt)
full_dt <- full_dt[, iteration := row_number(idx), by = c("triplet_id", "X4")]
full_dt <- full_dt[iteration == 1, ]
full_dt[, iteration := NULL]


correlation_two_entities <- function(
    n_thx_incl, filtercol, filterval, 
    group_cols, col1, col2, namecol
    ) {

  tbl_agreement <- tibble(
    full_dt[
      n >= n_thx_incl*2 & get(filtercol) %in% filterval, .(
        prop_lo = sum(X3 == id_lo) / .N,
        prop_mid = sum(X3 == id_mid) / .N,
        prop_hi = sum(X3 == id_hi) / .N,
        n_lo = sum(X3 == id_lo),
        n_mid = sum(X3 == id_mid),
        n_hi = sum(X3 == id_hi)
      ), by = group_cols]
  )
  
  # check that in each group at least n_thx_incl responses
  tbl_constrain <- tbl_agreement %>% 
    mutate(n_tot = n_lo + n_mid + n_hi) %>%
    filter(n_tot >= n_thx_incl)
  tbl_constrain <- tbl_constrain %>% count(triplet_id) %>% filter(n == 2) %>% select(-n)
  tbl_agreement <- tbl_constrain %>% left_join(tbl_agreement, by = c("triplet_id"))
  
  tbl_plt <- tbl_agreement %>%
    pivot_longer(cols = starts_with("prop_") | starts_with("n_")) %>%
    rename(measure_relpos = name, measure = value) %>%
    pivot_wider(names_from = {{ namecol }}, values_from = measure) %>%
    pivot_wider(names_from = measure_relpos, values_from = c({{ col1 }}, {{ col2 }}))
  
  tbl_plt[is.na(tbl_plt)] <- 0
  
  tbl_plt_long <- tbl_plt %>% select(c(triplet_id, contains("prop"), contains("_n_"))) %>%
    pivot_longer(-triplet_id) %>%
    rename(data_measure = name, val = value) %>%
    mutate(
      {{ namecol }} := str_extract(data_measure, "^[a-zA-Z1-2]*"),
      measuretype = str_match(data_measure, "_([a-z]*)_")[, 2]
    )
  
  tbl_plt_wide <- tbl_plt_long %>% #select(-measuretype) %>%
    mutate(
      location = str_extract(data_measure, "[a-z]*$"),
      location = factor(location, levels = c("lo", "mid", "hi"), labels = 1:3)
    ) %>%
    filter(location %in% c("1", "2")) %>%
    select(-data_measure) %>%
    pivot_wider(values_from = val, names_from = c({{ namecol }}, measuretype))
  
  return(tbl_plt_wide %>% arrange(triplet_id))
}


n_thx_incl <- 50
filtercol <- "dataset"
filterval <- c("Old", "New")
group_cols <- c("triplet_id", "dataset")
col1 <- "Old"
col2 <- "New"
namecol <- expr(dataset)

tbl_dataset <- correlation_two_entities(
  20, "dataset", c("Old", "New"), c("triplet_id", "dataset"), 
  "Old", "New", dataset
  )
tbl_half_old <- correlation_two_entities(
  20, "dataset", "Old", c("triplet_id", "half"), 
  "1", "2", half
  )
tbl_half_new <- correlation_two_entities(
  20, "dataset", "New", c("triplet_id", "half"), 
  "1", "2", half
  )

# now, calculate the correlation between the triplets shared across all three datasets

tbl_cor_joint_and_old <- tbl_half_old %>% 
  inner_join(
    tbl_dataset[, c("triplet_id", "location", "Old_prop", "New_prop")], 
    by = c("triplet_id", "location")
    )
tbl_cor_new <- tbl_half_new %>% 
  inner_join(tbl_cor_joint_and_old[, c("triplet_id", "location")], by = c("triplet_id", "location"))


tbl_avg_agreement <- tribble(
  ~Comparison, ~Correlation,
  "Old vs. New", cor(tbl_cor_joint_and_old$Old_prop, tbl_cor_joint_and_old$New_prop),
  "Old: Halves", cor(tbl_cor_joint_and_old$`1_prop`, tbl_cor_joint_and_old$`2_prop`),
  "New: Halves", cor(tbl_cor_new$`1_prop`, tbl_cor_new$`2_prop`)
) %>% mutate(
  prophecy = 2*Correlation/(1 + Correlation),
  correction = NA
)
tbl_avg_agreement[1, "prophecy"] <- NA
tbl_avg_agreement[1, "correction"] <- tbl_avg_agreement[1, "Correlation"] / sqrt(
  tbl_avg_agreement[2, "prophecy"] * tbl_avg_agreement[3, "prophecy"]
  )


saveRDS(tbl_avg_agreement, file = "data/dataframes-plotting-ms/tbl-avg-agreement.rds")
saveRDS(tbl_cor_joint_and_old, file = "data/dataframes-plotting-ms/tbl-cor-joint-and-old.rds")
saveRDS(tbl_cor_new, file = "data/dataframes-plotting-ms/tbl-cor-new.rds")

f_pl <- function(my_tbl, xvar, yvar, xlabel, ylabel, ttl, annot_string, annot_val) {
  ggplot(my_tbl, aes(!!sym(xvar), !!sym(yvar))) +
    geom_point(alpha = .25) +
    geom_abline(linewidth = 1) +
    theme_bw() +
    annotate("label", .3, .9, label = str_c(
      annot_string, " = ", round(annot_val, 2))
      ) +
    scale_x_continuous(breaks = seq(0, 1, by = .25), expand = c(0, 0.01)) +
    scale_y_continuous(breaks = seq(0, 1, by = .25), expand = c(0, 0.0)) +
    labs(x = xlabel, y = ylabel, title = ttl) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    theme(
      strip.background = element_rect(fill = "white"),
      text = element_text(size = 20),
      axis.text.x = element_text(angle = 90, vjust=.5),
      axis.text.y = element_text(hjust = .5)
      )
}

l_pl_cors <- pmap(
  list(
    list(tbl_cor_joint_and_old, tbl_cor_joint_and_old, tbl_cor_new), 
    c("Old_prop", "1_prop", "1_prop"), 
    c("New_prop", "2_prop", "2_prop"),
    c("Prop. Hebart et al. (2020)", "Prop. First Half", "Prop. First Half"),
    c("Prop. New Study", "Prop. Second Half", "Prop. Second Half"),
    c("Merged Data", "Hebart et al. (2020)", "New Study"),
    c("Corrected\nCorrelation", rep("Spearman-Brown\nCorrection", 2)),
    c(tbl_avg_agreement$correction[1], tbl_avg_agreement$prophecy[2:3])
  ), 
  f_pl
)


pl_consistencies <- arrangeGrob(l_pl_cors[[2]], l_pl_cors[[3]], l_pl_cors[[1]], ncol = 3)
grid.draw(pl_consistencies)
save_my_pdf(
  pl_consistencies, 
  "documents/writeup/figures-plotting/response-consistencies.pdf", 
  12, 4
  )

