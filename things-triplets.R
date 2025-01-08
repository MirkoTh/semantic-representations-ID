library(tidyverse)
library(rutils)
library(R.matlab)

things_dimension_labels <- R.matlab::readMat("data/labels.mat")
tbl_labels <- read_delim("data/unique_id.txt", delim = "\\", col_names = FALSE)
tbl_triplets <- read_delim("data/triplets_large_final_correctednc_correctedorder.csv")

grouped_agg(tbl_triplets, subject_id, RT) %>%
  summarize(m_RT = mean(mean_RT))

tbl_triplets %>% count(subject_id) %>% mutate(n_trials_cut = cut(n, c(seq(0, 1000, by = 100), Inf))) %>%
  count(n_trials_cut) %>%
  ggplot(aes(n_trials_cut, n)) + geom_col()


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
  tbl_ooo_train[1:100000, c("col_0", "col_1", "col_2")], 
  "data/train_90.txt", col_names = FALSE
)
write_delim(
  tbl_ooo_test[1:10000, c("col_0", "col_1", "col_2")],
  "data/test_10.txt", col_names = FALSE
)


# randomly sample subjects providing a substantial number of respo --------


# with added column denoting subject id
# but randomly select a set of participants instead of just obs
# first drop participants with < 400 obs
# then separate train and test data for every individual subject
thx <- 400
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

set.seed(8493)
s_random <- sample(unique(tbl_ooo_ID$subject_id), 75, replace = FALSE)
tbl_ooo_subset <- tbl_ooo_ID %>% filter(subject_id %in% s_random)
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

