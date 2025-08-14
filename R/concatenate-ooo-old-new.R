rm(list = ls())

library(tidyverse)

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
participant_ids_modeling_minus_old_ids <- unique(tbl_ooo_study1$X4)


max_id_things_train <- max(tbl_train_things$X4)
max_id_things_eval <- max(tbl_eval_things$X4)
assertthat::are_equal(max_id_things_train,  max_id_things_eval)


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
n_samples_old <- 10
set.seed(10)
tbl_testcase <- tbl_full[tbl_full$X4 %in% sample(max_id_things_train, n_samples_old), ]
tbl_testcase <- rbind(tbl_testcase, tbl_ooo_study1)


tbl_testcase <- rebase_subject_ids(tbl_testcase)

write_delim(
  tbl_testcase, 
  file = "data/study1-2025-08/ooo_data_modeling_old_and_new_testcase.txt", 
  col_names = FALSE
)


# only ids from new data set
unique_test_pids <- unique(tbl_testcase$X4)
pids_study1_testcase <- tibble(
  # ids used in modeling
  participant_id_model = unique_test_pids[which(unique_test_pids >= n_samples_old)],
  # ids only used for new study participants
  participant_id_new = participant_ids_new
)

# save the updated participant ids to easily extract data after model fitting
write_csv(
  pids_study1_testcase, 
  "data/study1-2025-08/new-participant-ids-in-joint-modeling-testcase.csv"
)


# Create and Save Half-Split Data Set ------------------------------------


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
