
#' Iterate over component directories to collect file paths
#'
#' @param study_dir Character. Name of the study directory (e.g., "study1").
#' @param comp_dir List of character vectors. Each element is a component directory name (e.g., list("comp1", "comp2")).
#'
#' @return Character vector. Full paths to all files within the specified component directories.
#'
iterate_files <- function(study_dir, comp_dir) {
  
  file_folders <- imap(comp_dir, function(cd, idx) {
    str_c(base_dir, study_dir, "/", cd, "/files/")
  })
  
  file_paths <- map(file_folders, function(x) str_c(list.dirs(x), dir(x)))
  
  return(file_paths)
}

#' Extract and categorize file paths from study directory
#'
#' @param base_dir Character. Path to the base directory containing study folders.
#'
#' @return Named list of character vectors. Each element contains file paths for a specific category:
#'   \itemize{
#'     \item{"cc"}{Comprehension check files}
#'     \item{"ooo"}{Odd-one-out task files}
#'     \item{"qs"}{Questionnaire files}
#'   }
#'
file_paths_separate <- function(base_dir) {
  study_dirs <- as.list(dir(base_dir))  # List of study subdirectories
  comp_dirs <- map(str_c(base_dir, study_dirs), dir)  # List of component subdirectories per study
  l_file_paths <- map2(study_dirs, comp_dirs, iterate_files)  # List of file paths per study
  
  v_file_paths <- reduce(l_file_paths, c) %>% reduce(c)  # Flatten into a single vector
  mask_cc <- str_detect(v_file_paths, "comprehension-check")
  mask_ooo <- str_detect(v_file_paths, "odd-one-out")
  mask_qs <- str_detect(v_file_paths, "questionnaires")
  
  v_cc_paths <- v_file_paths[mask_cc]
  v_ooo_paths <- v_file_paths[mask_ooo]
  v_qs_paths <- v_file_paths[mask_qs]
  
  return(list("cc" = v_cc_paths, "ooo" = v_ooo_paths, "qs" = v_qs_paths))
}

#' Replace participant IDs using a lookup table and prepare session metadata
#'
#' This function merges a data frame containing participant data (`tbl_df`) with a lookup table (`tbl_lookup`)
#' that maps original participant IDs (`prolific_pid`) to anonymized IDs (`participant_id_new`). It replaces
#' the original ID with the hashed version, adds a session ID, and reorders columns for consistency.
#'
#' @param tbl_df A data frame or tibble. Must contain a column named `participant_id` representing original IDs.
#' @param tbl_lookup A data frame or tibble. Must contain columns `prolific_pid` and `participant_id_new` for mapping.
#'
#' @return A tibble with:
#'   \itemize{
#'     \item `participant_id`: the anonymized ID replacing the original
#'     \item `session_id`: a constant value (1) added for session tracking
#'     \item All original columns from `tbl_df`, minus the original `participant_id`
#'   }
#'
#' @examples
#' tbl_df <- tibble(participant_id = c("A1", "B2"), score = c(10, 20))
#' tbl_lookup <- tibble(prolific_pid = c("A1", "B2"), participant_id_new = c("X100", "X200"))
#' hash_tbl(tbl_df, tbl_lookup)
#'
hash_tbl <- function(tbl_df, tbl_lookup) {
  tbl_new <- left_join(tbl_df, tbl_lookup, by = c("participant_id" = "prolific_pid"))
  tbl_new <- tbl_new %>% 
    select(-participant_id) %>% 
    rename(participant_id = participant_id_new) %>%
    relocate(participant_id, .before = 1) %>%
    mutate(session_id = 1) %>%
    relocate(session_id, .after = participant_id)
  return(tbl_new)
}

#' Format odd-one-out task data for modeling and further analysis
#'
#' This function transforms a tibble containing odd-one-out task responses into two outputs:
#' one for modeling and one for further analysis. It extracts stimulus IDs, identifies the odd
#' stimulus based on participant responses, and labels the remaining two stimuli as "positive"
#' and "negative". The modeling-ready output includes integer IDs for each stimulus role and
#' the participant ID, while the full output retains additional metadata and intermediate columns.
#'
#' @param tbl_ooo A tibble. Must contain:
#'   \itemize{
#'     \item `stimulus_ids`: a list-column of 3 stimulus IDs per trial
#'     \item `response`: an integer from 0 to 2 indicating which stimulus was chosen as the odd one
#'     \item `participant_id`: the participant's identifier
#'   }
#'
#' @return A named list with two tibbles:
#'   \itemize{
#'     \item `tbl_ooo_ids`: expanded and annotated version of the input, including stimulus positions and indices
#'     \item `tbl_ooo_ID_save`: modeling-ready tibble with columns `positive`, `negative`, `odd`, and `participant_id`
#'   }
#'
#' @examples
#' tbl_ooo <- tibble(
#'   participant_id = c("P1", "P2"),
#'   stimulus_ids = list(c("101", "102", "103"), c("201", "202", "203")),
#'   response = c(0, 2)
#' )
#' result <- ooo_modeling_format(tbl_ooo)
#' result$tbl_ooo_ID_save
#'
ooo_modeling_format <- function(tbl_ooo) {
  # format for modeling: anchor pos neg ID
  # saved as .txt
  tbl_ooo_ids <- tbl_ooo %>% unnest(stimulus_ids) %>%
    mutate(stimulus_loc = rep(c("ID1", "ID2", "ID3"), nrow(.) / 3)) %>%
    pivot_wider(names_from = stimulus_loc, values_from = stimulus_ids)
  
  tbl_ooo_ids <- tbl_ooo_ids %>% 
    rowwise() %>% 
    mutate(
      idx_odd = which(1:3 == response + 1),
      which_not_odd = list(c(1, 2, 3)[-idx_odd])
    ) %>% 
    unnest(which_not_odd) %>%
    mutate(idx_not_odd = rep(c("idx_positive", "idx_negative"), nrow(.) / 2)) %>%
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
  
  return(list(tbl_ooo_ids = tbl_ooo_ids, tbl_ooo_ID_save = tbl_ooo_ID_save))
}
