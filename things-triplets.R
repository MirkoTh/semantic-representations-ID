rm(list = ls())

library(tidyverse)
library(rutils)
library(R.matlab)

things_dimension_labels <- R.matlab::readMat("data/labels.mat")
things_words <- R.matlab::readMat("data/words.mat")

read_delim("data/data1854_batch5_test10.txt", col_names = FALSE)
tmp <- read_delim("data/spose_embedding_49d_sorted.txt", col_names = FALSE)

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






