
# Load Packages etc. ------------------------------------------------------

rm(list=ls())

library(tidyverse)

# home-grown
v_files <- c("R/utils.R")
walk(v_files, source)


# Load Data ---------------------------------------------------------------

tbl_cc <- read_csv("data/study1-2025-08/tbl_comprehension.csv")
tbl_ooo <- read_csv("data/study1-2025-08/tbl_ooo_ids.csv")
tbl_ooo_ID_save <- read_csv("data/study1-2025-08/tbl_ooo_ID_save.csv")
tbl_qs_num_long <- read_csv("data/study1-2025-08/tbl_qs_num_long.csv")
tbl_qs_txt <- read_csv("data/study1-2025-08/tbl_qs_txt.csv")
tbl_pids <- read_csv("data/study1-2025-08/tbl_participants.csv")


# exclusion criteria
thx_n_ooo_required <- 440
thx_n_ooo_streak_exclude <- 10 # same responses in a row
thx_ooo_rt_min_1 <- 800 #ms and prop responses below that thx
thx_prop_fast_1 <- .25
thx_ooo_rt_min_2 <- 1100 #ms
thx_prop_fast_2 <- .5
  
tbl_exclude <- tibble(
  participant_id = tbl_pids$participant_id_new
)

tbl_exclude <- tbl_exclude %>%
  left_join(tbl_cc %>% select(participant_id, n_attempts), by = "participant_id")

# exclude people with large number of comprehension check attempts
thx_attempts <- mean(tbl_cc$n_attempts) + 2*sd(tbl_cc$n_attempts)
tbl_exclude$exclude_attempts <- tbl_exclude$n_attempts > thx_attempts

tbl_exclude <- tbl_exclude %>% left_join(
  tbl_ooo %>% count(participant_id, name = "n_ooo") %>%
  mutate(exclude_ooo_trials = n_ooo < thx_n_ooo_required),
  by = "participant_id"
)


# exclusion criteria ooo:
# 1. more than n same ooo responses in a row
tbl_ooo_streak <- tbl_ooo %>%
  group_by(participant_id) %>%
  mutate(r_rep = as.numeric(response == lag(response, 1))) %>%
  replace_na(list(r_rep = 0)) %>%
  mutate(streak = cumsum_reset(r_rep) + 1) %>%
  summarize(max_response_streak = max(streak)) %>%
  ungroup()

tbl_exclude <- tbl_exclude %>% left_join(
  tbl_ooo_streak, by = "participant_id"
) %>% mutate(
  exclude_ooo_streak = max_response_streak > thx_n_ooo_streak_exclude
)

# 2. RTs too short
tbl_ooo_fast <- tbl_ooo %>% 
  mutate(rt_lo1 = rt < thx_ooo_rt_min_1, rt_lo2 = rt < thx_ooo_rt_min_2) %>%
  group_by(participant_id) %>% 
  summarize(n_trials = n(), n_lo1 = sum(rt_lo1), n_lo2 = sum(rt_lo2)) %>%
  ungroup() %>%
  mutate(prop_lo1 = n_lo1 / n_trials, prop_lo2 = n_lo2 / n_trials) %>%
  select(-c(n_trials, n_lo1, n_lo2))

tbl_exclude <- tbl_exclude %>% left_join(
  tbl_ooo_fast, by = "participant_id"
) %>% mutate(
  exclude_ooo_fast = prop_lo1 > thx_prop_fast_1 | prop_lo2 > thx_prop_fast_2
)

tbl_include <- tbl_exclude %>% 
  select(c(participant_id, starts_with("exclude"))) %>%
  pivot_longer(-participant_id) %>%
  group_by(participant_id) %>%
  summarize(exclude_eventually = sum(value)) %>%
  filter(exclude_eventually == 0) %>%
  select(participant_id)

write_csv(tbl_exclude, "data/study1-2025-08/tbl_exclude.csv")

cat(str_c("excluded: ", nrow(tbl_exclude) - nrow(tbl_include), " from ", nrow(tbl_exclude)))

tbl_ooo <- tbl_include %>% inner_join(tbl_ooo, by = "participant_id")
tbl_ooo_ID_save <- tbl_include %>% inner_join(tbl_ooo_ID_save, by = "participant_id")
tbl_qs_num_long <- tbl_include %>% inner_join(tbl_qs_num_long, by = "participant_id")
tbl_qs_txt <- tbl_include %>% inner_join(tbl_qs_txt, by = "participant_id")


# for further analysis
write_csv(tbl_ooo, "data/study1-2025-08/tbl_ooo_ids_excluded.csv")
write_csv(tbl_qs_num_long, "data/study1-2025-08/tbl_qs_num_long_excluded.csv")
write_csv(tbl_qs_txt, "data/study1-2025-08/tbl_qs_txt_excluded.csv")

# ooo file for pytorch model in python without colnames
tbl_ooo_ID_save <- tbl_ooo_ID_save %>%
  select(-session_id) %>%
  relocate(participant_id, .after = odd)

write_delim(
  tbl_ooo_ID_save, 
  file = "data/study1-2025-08/ooo_data_modeling_excluded.txt", 
  col_names = FALSE
)
