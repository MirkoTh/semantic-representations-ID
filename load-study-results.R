library(tidyverse)
library(jsonlite)



# Load Data ---------------------------------------------------------------

## comprehensions questions
tbl_comprehension <- jsonlite::fromJSON("data/study1-2025-08/comprehension-check.json") %>%
  as_tibble()


## odd-one-out
tbl_ooo <- jsonlite::fromJSON("data/study1-2025-08/odd-one-out.json") %>%
  as_tibble()


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

