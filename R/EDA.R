
# Load Packages etc. ------------------------------------------------------

rm(list=ls())

library(tidyverse)

# home-grown
v_files <- c("R/utils.R")
walk(v_files, source)


# Load Data ---------------------------------------------------------------

tbl_ooo <- read_csv("data/study1-2025-08/tbl_ooo_ids_excluded.csv")
tbl_qs_num_long <- read_csv("data/study1-2025-08/tbl_qs_num_long_excluded.csv")
tbl_qs_txt <- read_csv("data/study1-2025-08/tbl_qs_txt_excluded.csv")

n_participants <- length(unique(tbl_ooo$participant_id))
tbl_agreement <- tbl_ooo %>% group_by(triplet_id) %>% count(response) %>%
  mutate(prop = n / n_participants) %>% summarize(agreement = max(prop))


mn_agreement <- round(mean(tbl_agreement$agreement), 2)
ggplot(tbl_agreement, aes(agreement)) + 
  geom_vline(xintercept = .33, color = "red", linetype = "dotdash", alpha = .3, linewidth = 1) +
  geom_histogram(color = "black", fill = "skyblue2", aes(y = after_stat(count / sum(count)))) + 
  geom_label(y = .025, x = .15, label = str_c("Mean=", mn_agreement), size = 7) +
  geom_label(y = .05, x = .33, label = "Chance", size = 7, color="red", alpha = .3) +
  coord_cartesian(xlim = c(0, 1)) +
  theme_bw() +
  scale_x_continuous(expand = c(0.01, 0)) +
  scale_y_continuous(expand = c(0.01, 0)) +
  labs(x = "", y = "") + 
  theme(
    strip.background = element_rect(fill = "white"),
    text = element_text(size = 22),
    axis.text.x = element_text(angle = 45, hjust = 1)
    ) +
  labs(
    x = "Prop. Agreement",
    y = "Prop. Responses",
    title = "Agreement in Responses"
  )



# todos
# prepare correlational analyses between weights, etc. and demographics & questionnaires







