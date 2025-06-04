rm(list = ls())

library(tidyverse)
library(rutils)
library(R.matlab)

things_dimension_labels <- R.matlab::readMat("data/labels.mat")
things_words <- R.matlab::readMat("data/words.mat")

# read_delim("data/data1854_batch5_test10.txt", col_names = FALSE)
# tmp <- read_delim("data/spose_embedding_49d_sorted.txt", col_names = FALSE)

tbl_labels <- read_delim("data/unique_id.txt", delim = "\\", col_names = FALSE)
tbl_triplets <- read_delim("data/triplets_large_final_correctednc_correctedorder.csv")

grouped_agg(tbl_triplets, subject_id, RT) %>%
  summarize(m_RT = mean(mean_RT))

tbl_triplets %>% count(subject_id) %>% mutate(n_trials_cut = cut(n, c(seq(0, 1000, by = 100), Inf))) %>%
  count(n_trials_cut) %>%
  ggplot(aes(n_trials_cut, n)) + 
  geom_col(fill = "#6699FF", color = "black") +
  theme_bw() +
  scale_x_discrete(expand = c(0.01, 0)) +
  scale_y_continuous(expand = c(0.01, 0), breaks = seq(0, 9000, by = 1000)) +
  labs(x = "Nr. Trials", y = "Nr. Participants") +
  theme(
    strip.background = element_rect(fill = "white"),
    text = element_text(size = 22),
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.y = element_line(size = 1, color = "grey")  # Adjust the size to make gridlines thicker
  ) +
  scale_color_brewer(palette = "Set2", name = "") +
  scale_color_gradient2(name = "", high = "#66C2A5", low = "#FC8D62", mid = "white") +
  scale_color_gradient(name = "", high = "#66C2A5", low = "#FC8D62")


softmax <- function(x1, x2) {
  exp(x1) / (exp(x1) + exp(x2))
}



tbl_x <- tibble(x1 = seq(0, 6, length.out = 100))
tbl_x$x2 <- 3
tbl_x$y <- softmax(tbl_x$x1, tbl_x$x2)

ggplot(tbl_x, aes(x = x1 - x2, y = y)) +
  geom_line() +
  labs(title = "Softmax",
       x = expression(a %*% b),
       y = "Choice Probability") +
  theme_bw() +
  scale_x_continuous(expand = c(0.01, 0)) +
  scale_y_continuous(expand = c(0.01, 0)) +
  theme(
    strip.background = element_rect(fill = "white"), 
    text = element_text(size = 22),
    legend.position = "bottom",
    axis.text.x = element_blank()
  ) + 
  scale_color_brewer(palette = "Set2", name = "")


# save triplets to run representation model in python ---------------------


extract_anchor <- function(image1, image2, image3, OOO, a) {
  lgl <- c(image1, image2, image3) != OOO
  c(image1, image2, image3)[lgl][a]
}
tbl_ooo <- tbl_triplets %>%
  select(image1, image2, image3, choice, subject_id)
tbl_ooo$OOO <- pmap_dbl(tbl_ooo[, c("image1", "image2", "image3", "choice")], ~ c(..1, ..2, ..3)[..4])
tbl_ooo$anchor1 <- pmap_dbl(tbl_ooo[, c("image1", "image2", "image3", "OOO")], extract_anchor, a = 1)
tbl_ooo$anchor2 <- pmap_dbl(tbl_ooo[, c("image1", "image2", "image3", "OOO")], extract_anchor, a = 2)

tbl_ooo <- tbl_ooo %>%
  select(anchor1, anchor2, OOO, subject_id) %>%
  rename(
    col_0 = anchor1,
    col_1 = anchor2,
    col_2 = OOO
  ) %>%
  relocate(col_1, .before = col_2) %>% 
  relocate(col_0, .before = col_1) %>%
  mutate(
    col_0 = col_0 - 1,
    col_1 = col_1 - 1,
    col_2 = col_2 - 1
  )

rebase_subject_ids <- function(my_tbl) {
  my_tbl$subject_id <- factor(
    my_tbl$subject_id, 
    labels = seq(0, (length(unique(my_tbl$subject_id)) - 1))
  )
  my_tbl$subject_id <- as.numeric(my_tbl$subject_id) - 1
  return(my_tbl)
}
tbl_ooo <- rebase_subject_ids(tbl_ooo)



# random responses sampled from all subjects ------------------------------

set.seed(9879)
n_train <- round(nrow(tbl_ooo) * .9)
shuffled <- sample(nrow(tbl_ooo))
tbl_ooo_train <- tbl_ooo[shuffled[1:n_train],]
tbl_ooo_test <- tbl_ooo[shuffled[(n_train + 1):nrow(tbl_ooo)],]

tbl_ooo_train


write_delim(
  tbl_ooo_train[1:500000, c("col_0", "col_1", "col_2")], 
  "data/train_90.txt", col_names = FALSE
)
write_delim(
  tbl_ooo_test[1:50000, c("col_0", "col_1", "col_2")],
  "data/test_10.txt", col_names = FALSE
)


# randomly sample subjects providing a substantial number of respo --------


# with added column denoting subject id
# but randomly select a set of participants instead of just obs
# first drop participants with < 400 obs
# then separate train and test data for every individual subject
thx <- 250
prop_train <- .8
tbl_include <- tbl_ooo %>% count(subject_id) %>% filter(n >= thx) %>% select(-n)
tbl_ooo_ID <- tbl_ooo %>% inner_join(tbl_include, by = c("subject_id"))

cat(str_c(
  "from originally ", 
  format(length(unique(tbl_ooo$subject_id)), big.mark = "'"), 
  " subjects, reducing to ", 
  format(length(unique(tbl_ooo_ID$subject_id)), big.mark = "'"), 
  " subjects"
))



# bin n trials such that uniform sample along n trials can be selected
set.seed(8493)
tbl_ooo_ID_binned <- tbl_ooo_ID %>%
  group_by(subject_id) %>%
  summarize(n_trials = n()) %>% 
  mutate(n_trials_binned = cut(n_trials, seq(0, 100000, by = 400), labels = FALSE)) %>%
  group_by(n_trials_binned) %>%
  mutate(
    id_random = sample(n(), replace = FALSE)
  ) %>% ungroup() %>%
  filter(n_trials_binned <= 12)
tmp <- tbl_ooo_ID_binned %>% group_by(n_trials_binned) %>% count() %>% ungroup()
n_id_per_bin <- min(tmp$n)

#s_random <- sample(unique(tbl_ooo_ID$subject_id), 200, replace = FALSE)
#tbl_ooo_subset <- tbl_ooo_ID_binned %>% filter(subject_id %in% s_random)
tbl_ids <- tbl_ooo_ID_binned %>% filter(id_random < n_id_per_bin)
tbl_ooo_subset <- tbl_ooo_ID %>% inner_join(tbl_ids, by = "subject_id")

tbl_ooo_subset %>% group_by(subject_id) %>% summarize(n_trials = max(n_trials)) %>%
  ggplot(aes(n_trials)) +
  geom_histogram(breaks = seq(0, 5000, by = 400), fill = "#6699FF", color = "black") +
  theme_bw() +
  scale_x_continuous(expand = c(0.01, 0), breaks = seq(0, 5000, by = 500)) +
  scale_y_continuous(expand = c(0.01, 0)) +
  labs(x = "Nr. Trials", y = "Nr. Participants") +
  theme(
    strip.background = element_rect(fill = "white"),
    text = element_text(size = 22),
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# reset IDs to start from 0
tbl_ooo_subset <- rebase_subject_ids(tbl_ooo_subset)
tbl_ooo_subset <- tbl_ooo_subset %>%
  group_by(subject_id) %>%
  mutate(trial_id_random = sample(1:n())) %>%
  arrange(subject_id, trial_id_random) %>%
  mutate(
    cum_prop = trial_id_random / n()
  ) %>% select(-trial_id_random)
tbl_ooo_subset_train <- tbl_ooo_subset %>%
  filter(cum_prop <= prop_train) %>%
  select(-cum_prop)
tbl_ooo_subset_test <- tbl_ooo_subset %>%
  filter(cum_prop > prop_train) %>%
  select(-cum_prop)


tbl_old <- read_delim("data/test_10_ID_old.txt", col_names = FALSE)
tbl_old %>% count(X4) %>% mutate(n = 4*n) %>% arrange(n)

write_delim(
  tbl_ooo_subset_train %>% 
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/train_90_ID.txt", col_names = FALSE
)
write_delim(
  tbl_ooo_subset_test %>% 
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/test_10_ID.txt", col_names = FALSE
)


# small samples for testing
small_sample <- unique(tbl_ooo_subset$subject_id)[1:10]
write_delim(
  tbl_ooo_subset_train %>% 
    filter(subject_id %in% small_sample) %>%
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/train_90_ID_smallsample.txt", col_names = FALSE
)
write_delim(
  tbl_ooo_subset_test %>% 
    filter(subject_id %in% small_sample) %>%
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/test_10_ID_smallsample.txt", col_names = FALSE
)

# again full ID sample, but with trials shuffled across participants, i.e. shuffle out IDs
tbl_ooo_shuffle_id <- ungroup(tbl_ooo_subset)
set.seed(845872)
tbl_ooo_shuffle_id$subject_id_random <- sample(tbl_ooo_shuffle_id$subject_id)

prop_overlap <- sum(tbl_ooo_shuffle_id$subject_id_random == tbl_ooo_shuffle_id$subject_id) / nrow(tbl_ooo_shuffle_id)
tbl_unique_new_id <- tbl_ooo_shuffle_id %>%
  group_by(subject_id, subject_id_random) %>% 
  summarize(n_cross = n()) %>% group_by(subject_id) %>% 
  summarize(
    n_new_ids = length(unique(subject_id_random)),
    max_trials_same_subject_id = max(n_cross)
  )
tmp <- tbl_ooo_shuffle_id %>% count(subject_id)
tmp2 <- tmp %>% left_join(tbl_unique_new_id, by = "subject_id") %>% arrange(desc(max_trials_same_subject_id))
tmp2$n_per_subject <- tmp2$n / tmp2$n_new_ids

cat(str_c(
  "proportion overlap in subject ids old vs. new = ", round(prop_overlap, 3), "\n",
  "maximally ", round(max(tmp2$max_trials_same_subject_id), 1), " trials from the same subject per new ID,\n",
  "which refers to a proportion of: ", tmp2$max_trials_same_subject_id[1] / tmp2$n[1]
))
tmp2 %>% arrange(desc(max_trials_same_subject_id))

tbl_ooo_shuffle_id <- tbl_ooo_shuffle_id %>%
  select(c(col_0, col_1, col_2, subject_id_random)) %>%
  rename(subject_id = subject_id_random) %>%
  arrange(subject_id)


tbl_ooo_shuffle_id <- rebase_subject_ids(tbl_ooo_shuffle_id)
tbl_ooo_shuffle_id <- tbl_ooo_shuffle_id %>%
  group_by(subject_id) %>%
  mutate(trial_id_random = sample(1:n())) %>%
  arrange(subject_id, trial_id_random) %>%
  mutate(
    cum_prop = trial_id_random / n()
  ) %>% select(-trial_id_random)
tbl_ooo_shuffle_id_train <- tbl_ooo_shuffle_id %>%
  filter(cum_prop <= prop_train) %>%
  select(-cum_prop)
tbl_ooo_shuffle_id_test <- tbl_ooo_shuffle_id %>%
  filter(cum_prop > prop_train) %>%
  select(-cum_prop)

write_delim(
  tbl_ooo_shuffle_id_train %>% 
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/train_shuffled_90_ID.txt", col_names = FALSE
)
write_delim(
  tbl_ooo_shuffle_id_test %>% 
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/test_shuffled_10_ID.txt", col_names = FALSE
)


# extract agreement among triplets ----------------------------------------


t_start <- Sys.time()
tbl_count_triplets <- tbl_ooo %>%
  #slice_sample(n = 500000) %>%
  mutate(incr = 1:nrow(.)) %>%
  rowwise() %>%
  mutate(
    id_lo = min(c(col_0, col_1, col_2)),
    id_hi = max(c(col_0, col_1, col_2)),
    id_mid = c(col_0, col_1, col_2)[!c(col_0, col_1, col_2) %in% c(id_lo, id_hi)]
  ) %>%
  relocate(id_mid, .before = id_hi) %>%
  group_by(id_lo, id_mid, id_hi) %>%
  mutate(rwn = row_number(incr)) %>%
  mutate(
    n_encounters = max(rwn)
  ) %>%
  arrange(desc(rwn)) %>%
  ungroup()

t_end <- Sys.time()
t_duration <- t_end - t_start
cat(t_duration)

tbl_subset_items <- tbl_count_triplets %>% filter(n_encounters >= 10) %>% mutate(triplet_id = factor(str_c(id_lo, id_mid, id_hi)))
tbl_subset_items$triplet_id <- factor(tbl_subset_items$triplet_id, labels = 1:length(unique(tbl_subset_items$triplet_id)))
tbl_subset_items %>% count(triplet_id) %>% ggplot(aes(n)) + geom_histogram(color = "white", fill = "#6699FF") + coord_cartesian(xlim = c(0, 150))
tbl_agreement <- tbl_subset_items %>% 
  group_by(triplet_id) %>%
  summarize(
    n_0 = sum(col_0 == id_lo),
    n_1 = sum(col_0 == id_mid),
    n_2 = sum(col_0 == id_hi)
    ) %>% ungroup() %>%
  rowwise() %>%
  mutate(
    n_min = min(c(n_0, n_1, n_2)),
    n_max = max(c(n_0, n_1, n_2)),
    n_med = (n_0 + n_1 + n_2) - (n_min + n_max),
    prop_max = n_max/(n_min + n_med + n_max)
  ) %>% ungroup()

ggplot(tbl_agreement, aes(prop_max)) +
  geom_histogram()

tbl_agreement %>% mutate(prop_max_weighted = prop_max * (n_min + n_med + n_max)) %>% ungroup() %>% summarize(n_tot = sum(n_0 + n_1 + n_2), n_agree = sum(prop_max_weighted)) %>%
  mutate(avg_agreement = n_agree/n_tot)


tbl_diagnostic_items <- tbl_subset_items %>% group_by(id_lo, id_mid, id_hi) %>% count() %>% ungroup()
write_csv(tbl_diagnostic_items, file = "diagnostic-triplets.csv")


# Data Selection for Modeling Item Difficulties ---------------------------


# then calculate number of trials for all these participants

# select those participants with substantial number of trials (e.g., 250 trials)
thx <- 250
prop_train <- .8
tbl_include_trials <- tbl_ooo %>% count(subject_id) %>% filter(n >= thx) %>% select(-n)
# select all participants who have contributed to pairs with substantial number of trials (e.g., 10/20)
# because only for those we can test whether modeling IDs improves particularly low-agreement pairs
tbl_include_items <- tibble(subject_id = sort(unique(tbl_subset_items$subject_id)))
# take the intersection
tbl_include_both <- tbl_include_trials %>% inner_join(tbl_include_items, by = "subject_id")
tbl_ooo_ID_item <- tbl_ooo %>% inner_join(tbl_include_both, by = "subject_id")


# reset IDs to start from 0
tbl_ooo_ID_item <- rebase_subject_ids(tbl_ooo_ID_item)
tbl_ooo_ID_item <- tbl_ooo_ID_item %>%
  group_by(subject_id) %>%
  mutate(trial_id_random = sample(1:n())) %>%
  arrange(subject_id, trial_id_random) %>%
  mutate(
    cum_prop = trial_id_random / n()
  ) %>% select(-trial_id_random)
tbl_ooo_ID_item_subset_train <- tbl_ooo_ID_item %>%
  filter(cum_prop <= prop_train) %>%
  select(-cum_prop)
tbl_ooo_ID_item_subset_test <- tbl_ooo_ID_item %>%
  filter(cum_prop > prop_train) %>%
  select(-cum_prop)

write_delim(
  tbl_ooo_ID_item_subset_train %>% 
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/train_90_ID_item.txt", col_names = FALSE
)
write_delim(
  tbl_ooo_ID_item_subset_test %>% 
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/test_10_ID_item.txt", col_names = FALSE
)

# and also shuffle across participants

tbl_ooo_ID_item_shuffle_id <- ungroup(tbl_ooo_ID_item)
set.seed(845872)
tbl_ooo_ID_item_shuffle_id$subject_id_random <- sample(tbl_ooo_ID_item_shuffle_id$subject_id)

tbl_ooo_ID_item_shuffle_id <- tbl_ooo_ID_item_shuffle_id %>%
  select(c(col_0, col_1, col_2, subject_id_random)) %>%
  rename(subject_id = subject_id_random) %>%
  arrange(subject_id)


tbl_ooo_ID_item_shuffle_id <- rebase_subject_ids(tbl_ooo_ID_item_shuffle_id)
tbl_ooo_ID_item_shuffle_id <- tbl_ooo_ID_item_shuffle_id %>%
  group_by(subject_id) %>%
  mutate(trial_id_random = sample(1:n())) %>%
  arrange(subject_id, trial_id_random) %>%
  mutate(
    cum_prop = trial_id_random / n()
  ) %>% select(-trial_id_random)
tbl_ooo_ID_item_shuffle_id_train <- tbl_ooo_ID_item_shuffle_id %>%
  filter(cum_prop <= prop_train) %>%
  select(-cum_prop)
tbl_ooo_ID_item_shuffle_id_test <- tbl_ooo_ID_item_shuffle_id %>%
  filter(cum_prop > prop_train) %>%
  select(-cum_prop)

write_delim(
  tbl_ooo_ID_item_shuffle_id_train %>% 
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/train_shuffled_90_ID_item.txt", col_names = FALSE
)
write_delim(
  tbl_ooo_ID_item_shuffle_id_test %>% 
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/test_shuffled_10_ID_item.txt", col_names = FALSE
)








# Data Preparation for Split-Half Reliability Analyses --------------------

# 
tbl_ooo_ID_item %>% count(subject_id) %>% ggplot(aes(n)) + geom_histogram(binwidth = 250) + coord_cartesian(xlim = c(0, 10000))


# cumprop_cut_lag gives us 10% buckets of participants.
# this will allow us to analyze split-half reliability with varying numbers of responses

tbl_bins_samesize <- tbl_ooo_ID_item %>%
  count(subject_id) %>% 
  ungroup() %>% 
  arrange(n)%>% 
  group_by(n) %>% 
  count(name = "n_per_ntrials") %>% 
  ungroup() %>% 
  mutate(n_tot = sum(n_per_ntrials), prop = n_per_ntrials / n_tot) %>%
  ungroup() %>%
  mutate(
    cumprop = cumsum(prop), 
    cumprop_cut = cut(cumprop, seq(0, 1, by = .1)), 
    cumprop_cut_lag = lag(cumprop_cut),
    cumprop_cut_lag = coalesce(cumprop_cut_lag, cumprop_cut)
    )
tbl_bins_samesize$cumprop_cut_lag <- factor(tbl_bins_samesize$cumprop_cut_lag, labels = 1:10)

# cum_prop is a randomly arranged series of trials, so we can just split at 50%
tbl_split_half <- tbl_ooo_ID_item %>% group_by(subject_id) %>% mutate(n = n()) %>%
  left_join(tbl_bins_samesize %>% select(n, cumprop_cut_lag), by = "n") %>%
  ungroup() %>%
  mutate(half = factor(cum_prop <= .5, labels = c(1, 2)))

# save the two halfs to run the model upon
write_delim(
  tbl_split_half %>% 
    filter(half == 1) %>%
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/train_shuffled_90_ID_item.txt", col_names = FALSE
)
write_delim(
  tbl_split_half %>% 
    filter(half == 2) %>%
    select(c("col_0", "col_1", "col_2", "subject_id")), 
  "data/test_shuffled_10_ID_item.txt", col_names = FALSE
)

# save a lookup table mapping subject_ids to 10% buckets
tbl_bucket_lookup <- tbl_split_half %>% group_by(subject_id) %>% 
  summarize(cumprop_cut_lag = unique(cumprop_cut_lag)) %>%
  ungroup()

write_delim(tbl_bucket_lookup, "data/splithalf_lookup_bucket.txt", colnames = FALSE)


















colnames(tbl_old) <- c("col_0", "col_1", "col_2", "subject_id")
tmp <- tbl_old %>% 
  left_join(tbl_ooo, by = c("col_0", "col_1", "col_2"), suffix = c("_old", "_new"), relationship = "many-to-many") %>%
  group_by(subject_id_old, subject_id_new) %>%
  count() %>%
  group_by(subject_id_old) %>%
  filter(n == max(n))



tbl_train_old <- read_delim("data/train_90_ID_old.txt", col_names = c("anchor", "positive", "negative", "ID"))
tbl_test_old <- read_delim("data/test_10_ID_old.txt", col_names = c("anchor", "positive", "negative", "ID"))

tbl_train_new <- read_delim("data/train_90_ID.txt", col_names = c("anchor", "positive", "negative", "ID"))
tbl_test_new <- read_delim("data/test_10_ID.txt", col_names = c("anchor", "positive", "negative", "ID"))


tbl_train_new_old <- tbl_train_new %>% left_join(tbl_train_old, by = c("anchor", "positive", "negative"), relationship = "many-to-many")
tbl_train_new_old %>% filter(ID.x == ID.y) %>% count()
tbl_test_new_old <- tbl_test_new %>% left_join(tbl_test_old, by = c("anchor", "positive", "negative"), relationship = "many-to-many")
tbl_test_new_old %>% filter(ID.x == ID.y) %>% count()



tbl_train_shuffled <- read_delim("data/train_shuffled_90_ID.txt", col_names = c("anchor", "positive", "negative", "ID"))
tbl_train_new_shuffled <- tbl_train_new %>% left_join(tbl_train_shuffled, by = c("anchor", "positive", "negative"), relationship = "many-to-many")
tbl_train_new_shuffled %>% filter(ID.x == ID.y) %>% count()

