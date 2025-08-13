# load study results into data containers
# hash prolific ids and save lookup table locally

rm(list = ls())

library(tidyverse)
library(jsonlite)

# home-grown
l_load <- c("R/utils.R")
walk(l_load, source)

# recode reversely coded questionnaire items
# create scales on questionnaire responses


# first, list all result files available in the respective folders

# Set the base directory containing study data
base_dir <- "data/study1-2025-08/jatos_results_files_20250813071123/"
l_paths_sep <- file_paths_separate(base_dir)



# Load Data ---------------------------------------------------------------

## comprehensions questions
tbl_comprehension <- map(l_paths_sep$cc, function(x) as_tibble(fromJSON(x))) %>% reduce(rbind)
# runthrough time is time after questions have been answered correctly - time when first ooo page was displayed in ms
tbl_comprehension <- tbl_comprehension %>%
  mutate(t_comprehension_min = t_comprehension / 1000 / 60)


## odd-one-out
tbl_ooo <- map(l_paths_sep$ooo, function(x) as_tibble(fromJSON(x))) %>% reduce(rbind)
l_ooo <- ooo_modeling_format(tbl_ooo)
tbl_ooo_ID_save <- l_ooo$tbl_ooo_ID_save
tbl_ooo_ids <- l_ooo$tbl_ooo_ids

## questionnaires
cols_txt <- c("workHistory", "interests1", "interests2", "interests3", "feedback")
cols_id <- c("session_id", "participant_id")

tbl_qs_prep <- map(l_paths_sep$qs, function(x) as_tibble(fromJSON(x))) %>% reduce(rbind)


# numeric responses from questionnaires
tbl_qs_num <- tbl_qs_prep %>%
  select(-all_of(c(cols_txt)))
# control data types
cols_numeric <- colnames(tbl_qs_num)[!colnames(tbl_qs_num) %in% cols_id]
tbl_qs_num[, cols_numeric] <- map(tbl_qs_num[, cols_numeric], as.numeric)
tbl_qs_num_long <- tbl_qs_num %>% pivot_longer(cols=-all_of(cols_id))

# text responses from questionnaires
tbl_qs_txt <- tbl_qs_prep %>%
  select(all_of(c(cols_id, cols_txt)))



# Hash IDs ----------------------------------------------------------------

# files to hash and save:
l_tbl_to_hash <- list(
  tbl_comprehension = tbl_comprehension, 
  tbl_ooo_ids = tbl_ooo_ids, 
  tbl_ooo_ID_save = tbl_ooo_ID_save, 
  tbl_qs_num_long = tbl_qs_num_long, 
  tbl_qs_txt = tbl_qs_txt
)

tbl_lookup <- tibble(
  prolific_pid = unique(c(
    tbl_comprehension$participant_id,
    tbl_ooo_ids$participant_id,
    tbl_qs_num_long$participant_id,
    tbl_qs_txt$participant_id
  )),
  participant_id_new = 1:length(prolific_pid)
)

l_tbl_hashed <- map(l_tbl_to_hash, hash_tbl, tbl_lookup = tbl_lookup)


list2env(l_tbl_hashed, rlang::current_env())


# Save Files ---------------------------------------------------------


# lookup table
write_csv(tbl_lookup, file = "data/study1-2025-08/tbl_lookup.csv")
write_csv(tbl_lookup %>% select(participant_id_new), file = "data/study1-2025-08/tbl_participants.csv")

# data for analysis
pths <- str_c("data/study1-2025-08/", names(l_tbl_hashed), ".csv")
walk2(l_tbl_hashed, pths, write_csv)
