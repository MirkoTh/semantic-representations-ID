
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



ggplot(tbl_agreement, aes(agreement)) + 
  geom_vline(xintercept = .33, color = "red", linetype = "dotdash", alpha = .3, linewidth = 1) +
  geom_histogram(color = "black", fill = "skyblue2", aes(y = after_stat(count / sum(count)))) + 
  geom_label(y = .1, x = .15, label = str_c("Mean=", round(mean(tbl_agreement$agreement), 2)), size = 7) +
  geom_label(y = .15, x = .33, label = "Chance", size = 7, color="red", alpha = .3) +
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




# save average RT for the triplet task
tbl_avg <- tbl_ooo %>% group_by(participant_id) %>%
  summarize(mn_rt = mean(rt)) %>% ungroup()
tbl_pid_mapping <- read_csv("data/study1-2025-08/new-participant-ids-in-joint-modeling.csv")
tbl_avg <- tbl_avg %>% left_join(tbl_pid_mapping, by = c("participant_id" = "participant_id_new"))
write_csv(tbl_avg, "data/study1-2025-08/avg-rt.csv")


