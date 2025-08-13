library(tidyverse)


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
)

max_id_things_train <- max(tbl_train_things$X4)
max_id_things_eval <- max(tbl_eval_things$X4)
assertthat::are_equal(max_id_things_train,  max_id_things_eval)

tbl_ooo_study1$X4 <- tbl_ooo_study1$X4 + max_id_things_train
pids_study1 <- tibble(participant_pid = unique(tbl_ooo_study1$X4))

# save the updated participant ids to easily extract data after model fitting
write_csv(pids_study1, "data/study1-2025-08/new-participant-ids-in-joint-modeling.csv")


# full data set
tbl_full <- rbind(tbl_train_things, tbl_eval_things, tbl_ooo_study1)

write_delim(
  tbl_full, 
  file = "data/study1-2025-08/ooo_data_modeling_old_and_new.txt", 
  col_names = FALSE
)

# small data set for testing

set.seed(10)
tbl_testcase <- tbl_full[tbl_full$X4 %in% sample(unique(tbl_full$X4), 10), ]

write_delim(
  tbl_testcase, 
  file = "data/study1-2025-08/ooo_data_modeling_old_and_new_testcase.txt", 
  col_names = FALSE
) 
