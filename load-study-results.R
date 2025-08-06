library(tidyverse)
library(jsonlite)



# Load Data ---------------------------------------------------------------

## comprehensions questions
tbl_comprehension <- jsonlite::fromJSON("data/study1-2025-08/comprehension-check.json") %>%
  as_tibble()


## odd-one-out
tbl_ooo <- jsonlite::fromJSON("data/study1-2025-08/odd-one-out.json") %>%
  as_tibble()

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
write_delim(
  tbl_ooo_ID_save, 
  file = "data/study1-2025-08/ooo-data-modeling.txt", 
  col_names = FALSE
)

## questionnaires
cols_separate <- c("workHistory", "interests1", "interests2", "interests3", "feedback")
qs_prep <- jsonlite::fromJSON("data/study1-2025-08/questionnaires.json") %>% 
  as_tibble()

# numeric responses from questionnaires
tbl_qs_num <- qs_prep %>%
  select(-all_of(cols_separate))
# control data types
cols_character <- "participant_id"
cols_numeric <- colnames(tbl_qs_num)[!colnames(tbl_qs_num) %in% cols_character]
tbl_qs_num[, cols_numeric] <- map(tbl_qs_num[, cols_numeric], as.numeric)
tbl_qs_num %>% pivot_longer(cols=-participant_id)

# text responses from questionnaires
tbl_qs_txt <- qs_prep %>%
  select(all_of(c("participant_id", cols_separate)))

