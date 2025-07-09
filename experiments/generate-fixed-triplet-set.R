# considerations
# create a set consisting of 440 different triplets
# 220 triplets are taken from the set with the largest delta between avg and ID model
# 220 triplets are created from the full set of 1854 objects by sampling randomly
## here, check that categories are used approximately equally often as in base distribution

tbl_categories <- read_tsv("data/category27_manual.tsv")
tbl_labels <- read_delim("data/unique_id.txt", delim = "\\", col_names = FALSE)
tbl_triplets <- read_delim("data/triplets_large_final_correctednc_correctedorder.csv")

cbind(tbl_labels, tbl_categories) %>%
  pivot_longer(-X1) %>%
  filter(value == 1) %>%
  count(X1) %>%
  arrange(desc(n))

cbind(tbl_labels, tbl_categories) %>%
  pivot_longer(-X1) %>%
  filter(value == 1) %>%
  count(name) %>%
  arrange(desc(n)) %>%
  mutate(name = fct_inorder(name)) %>%
  ggplot(aes(name, n)) + geom_col() +
  theme(axis.text.x = element_text(angle = 45))
