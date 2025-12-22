library(tidyverse)
library(grid)

source("R/utils.R")



# Nr. Participants per Nr. Trials -----------------------------------------


tbl_triplets <- read_delim("data/triplets_large_final_correctednc_correctedorder.csv")

tbl_trials_hebart <- tbl_triplets %>% count(subject_id) %>%
  mutate(n_trials_cut = cut(
    n, c(seq(0, 600, by = 20), Inf), 
    labels = c(seq(20, 600, by = 20), ">600")
  )
  ) %>%
  count(n_trials_cut)

tbl_triplets  %>% count(subject_id) %>% arrange(desc(n))
tbl_triplets  %>% count(subject_id) %>% filter(n >= 260) %>% count(subject_id)

tbl_trials_hebart$n_trials_cut <- fct_inorder(factor(tbl_trials_hebart$n_trials_cut))


tbl_agg <- tbl_triplets %>% 
  count(subject_id) %>%
  mutate(n = ifelse(n >= 600, 600, n))

tbl_agg_agg <- tbl_agg %>% count(nn = n)

pl_ntrials_hebart <- ggplot(tbl_agg, aes(n)) +
  geom_vline(xintercept = 259, color = "darkred", linewidth = 1) +
  annotate("rect",
           xmin = 260, xmax = 620,
           ymin = 0, ymax = 5000,
           fill = "seagreen3", alpha = 0.5) +
  geom_histogram(color = "black", fill = "dodgerblue1", binwidth = 20) +
  annotate("label", x = 160, y = 2900,
           label = "Participants\nwith < 260 trials\nwere excluded",
           size = 6) +
  geom_label(data = tbl_agg_agg, aes(nn, n + 400, label = n), angle = 90, size = 6) +#%>% filter(nn %% 100 == 0)
  theme_bw() +
  scale_x_continuous(expand = c(0, 0.02), breaks = seq(40, 600, by = 40), labels = c(seq(40, 580, by = 40), ">=600")) +
  scale_y_continuous(expand = c(.035, 0.02)) +
  labs(x = "Number of completed trials", y = "Number of participants") +
  theme(
    strip.background = element_rect(fill = "white"),
    text = element_text(size = 22),
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.y = element_line(size = 1, color = "grey")  # Adjust the size to make gridlines thicker
  )


ggsave("documents/writeup/figures-plotting/plot-ntrials-hebart.png",
       plot = pl_ntrials_hebart,
       width = 7, height = 5)



# Dimensional Weights -----------------------------------------------------

ndims <- 7

set.seed(23)
os_mat <- map(rep(ndims, 3), runif,min = 0, max = 2) %>% 
  reduce(rbind) %>% as.data.frame() %>% as_tibble()
os <- os_mat %>%
  mutate(id = 1:3) %>% pivot_longer(-id)
os$name <- as.numeric(factor(os$name, labels = 1:ndims))
ws_mat <- map(rep(ndims, 2), runif, min = 0, max = 2) %>% 
  reduce(rbind) %>% as.data.frame() %>% as_tibble()
ws <- ws_mat %>%
  mutate(id = 1:2) %>% pivot_longer(-id)
ws$name <- as.numeric(factor(ws$name, labels = 1:ndims))

plot_vector <- function(idx, my_tbl, flip_coords = "nothing") {
  pl <- ggplot(my_tbl %>% filter(id == idx), aes("", name)) +
    geom_tile(aes(fill = value)) +
    geom_label(aes(label = round(value, 2))) +
    labs(x = "", y = "Dimension") +
    scale_y_continuous(breaks = 1:7, expand = c(0, 0)) +
    theme_bw() +
    theme(
      strip.background = element_rect(fill = "white"),
      text = element_text(size = 22),
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major.y = element_line(size = 1, color = "grey")  # Adjust the size to make gridlines thicker
    ) +
    scale_fill_gradient2(low = "darkred", mid = "white", high = "darkgreen", midpoint = 1, guide = "none")
  
  if (flip_coords == "x_and_y") {
    pl <- pl + coord_flip()
  }
  if (flip_coords == "revert_y") {
    pl <- pl + scale_y_reverse(breaks = 1:7, expand = c(0, 0))
  }
  
  return(pl)
}

l_plot_os <- map(1:3, plot_vector, my_tbl = os, flip_coords = "x_and_y")
l_plot_ws <- map(1:2, plot_vector, my_tbl = ws, flip_coords = "revert_y")

pths_os <- str_c("documents/writeup/figures-plotting/", c("o1.pdf", "o2.pdf", "o3.pdf"))
map2(l_plot_os, pths_os, save_my_pdf, w = 4.25, h = 1.4)


pths_ws <- str_c("documents/writeup/figures-plotting/", c("w1.pdf", "w2.pdf"))
map2(l_plot_ws, pths_ws, save_my_pdf, w = 1.25, h = 4.5)


os_mat <- matrix(as.matrix(os_mat), nrow = 3)
ws1_mat <- matrix(reduce(rep(ws_mat[1, ], 3), c), nrow = 3, byrow = TRUE)
ws2_mat <- matrix(reduce(rep(ws_mat[2, ], 3), c), nrow = 3, byrow = TRUE)

ow_1_mat <- os_mat * ws1_mat
ow_1 <- as.data.frame(ow_1_mat) %>% as_tibble() %>%
  mutate(id = 1:3) %>%
  pivot_longer(-id)
ow_1$name <- as.numeric(factor(ow_1$name, labels = 1:ndims))

ow_2_mat <- os_mat * ws2_mat
ow_2 <- as.data.frame(os_2_mat) %>% as_tibble() %>%
  mutate(id = 1:3) %>%
  pivot_longer(-id)
ow_2$name <- as.numeric(factor(ow_2$name, labels = 1:ndims))


l_plot_ow_1 <- map(1:3, plot_vector, my_tbl = ow_1, flip_coords = "x_and_y")
l_plot_ow_2 <- map(1:3, plot_vector, my_tbl = ow_2, flip_coords = "x_and_y")

pths_ow_1 <- str_c("documents/writeup/figures-plotting/", c("ow_1_1.pdf", "ow_2_1.pdf", "ow_3_1.pdf"))
pths_ow_2 <- str_c("documents/writeup/figures-plotting/", c("ow_1_2.pdf", "ow_2_2.pdf", "ow_3_2.pdf"))
map2(l_plot_ow_1, pths_ow_1, save_my_pdf, w = 4.25, h = 1.4)
map2(l_plot_ow_2, pths_ow_2, save_my_pdf, w = 4.25, h = 1.4)

ow_1_mat

dot_prod <- function(x, y) {
  sum(x * y)
}

dot_prod <- function(idx1, idx2, my_mat) {
  sum(my_mat[idx1, ] * my_mat[idx2, ])
}

logits1 <- map2_dbl(c(1, 1, 2), c(2, 3, 3), dot_prod, my_mat = ow_1_mat)
logits2 <- map2_dbl(c(1, 1, 2), c(2, 3, 3), dot_prod, my_mat = ow_2_mat)
prob1 <- round(logits1/sum(logits1), 2)
prob2 <- round(logits2/sum(logits2), 2)


