rm(list = ls())

library(tidyverse)
library(jsonlite)

# home-grown
l_load <- c("utils.R")
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
## odd-one-out
tbl_ooo <- map(l_paths_sep$ooo, function(x) as_tibble(fromJSON(x))) %>% reduce(rbind)



# format for modeling: anchor pos neg ID
# saved as .txt
tbl_ooo_ids <- tbl_ooo %>% unnest(stimulus_ids) %>%
  mutate(stimulus_loc = rep(c("ID1", "ID2", "ID3"), nrow(.)/3)) %>%
  pivot_wider(names_from = stimulus_loc, values_from = stimulus_ids)
tbl_ooo_ids <- tbl_ooo_ids %>% rowwise() %>% mutate(
  idx_odd = which(1:3 == response + 1),
  which_not_odd = list(c(1, 2, 3)[-idx_odd])
) %>% unnest(which_not_odd) %>%
  mutate(idx_not_odd = rep(c("idx_positive", "idx_negative"), nrow(.)/2)) %>%
  pivot_wider(names_from = idx_not_odd, values_from = which_not_odd) %>%
  relocate(idx_positive, .before = idx_odd) %>%
  relocate(idx_negative, .before = idx_odd)

tbl_ooo_ID_save <- tbl_ooo_ids %>% 
  rowwise() %>%
  mutate(
    positive = c(ID1, ID2, ID3)[idx_positive],
    negative = c(ID1, ID2, ID3)[idx_negative],
    odd = c(ID1, ID2, ID3)[idx_odd]
  ) %>%
  select(positive, negative, odd, participant_id) %>%
  mutate(
    positive = as.integer(positive),
    negative = as.integer(negative),
    odd = as.integer(odd)
  )
tbl_ooo_ID_save$participant_id <- factor(
  tbl_ooo_ID_save$participant_id, 
  labels = 1:length(unique(tbl_ooo_ID_save$participant_id))
  )

write_delim(
  tbl_ooo_ID_save, 
  file = "data/study1-2025-08/ooo-data-modeling.txt", 
  col_names = FALSE
)

## questionnaires
cols_separate <- c("workHistory", "interests1", "interests2", "interests3", "feedback")

tbl_qs_prep <- map(l_paths_sep$qs, function(x) as_tibble(fromJSON(x))) %>% reduce(rbind)




# numeric responses from questionnaires
tbl_qs_num <- tbl_qs_prep %>%
  select(-all_of(cols_separate))
# control data types
cols_character <- c("session_id", "participant_id")
cols_numeric <- colnames(tbl_qs_num)[!colnames(tbl_qs_num) %in% cols_character]
tbl_qs_num[, cols_numeric] <- map(tbl_qs_num[, cols_numeric], as.numeric)
tbl_qs_num %>% pivot_longer(cols=-participant_id)

# text responses from questionnaires
tbl_qs_txt <- qs_prep %>%
  select(all_of(c("participant_id", cols_separate)))

