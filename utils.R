
#' Iterate over component directories to collect file paths
#'
#' @param study_dir Character. Name of the study directory (e.g., "study1").
#' @param comp_dir List of character vectors. Each element is a component directory name (e.g., list("comp1", "comp2")).
#'
#' @return Character vector. Full paths to all files within the specified component directories.
#'
iterate_files <- function(study_dir, comp_dir) {
  file_dir_1 <- str_c(base_dir, study_dir, "/", comp_dir[[1]], "/files/")
  file_dir_2 <- str_c(base_dir, study_dir, "/", comp_dir[[2]], "/files/")
  
  return(c(
    str_c(list.dirs(file_dir_1), dir(file_dir_1)),
    str_c(list.dirs(file_dir_2), dir(file_dir_2))
  ))
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
  
  v_file_paths <- reduce(l_file_paths, c)  # Flatten into a single vector
  mask_cc <- str_detect(v_file_paths, "comprehension-check")
  mask_ooo <- str_detect(v_file_paths, "odd-one-out")
  mask_qs <- str_detect(v_file_paths, "questionnaires")
  
  v_cc_paths <- v_file_paths[mask_cc]
  v_ooo_paths <- v_file_paths[mask_ooo]
  v_qs_paths <- v_file_paths[mask_qs]
  
  return(list("cc" = v_cc_paths, "ooo" = v_ooo_paths, "qs" = v_qs_paths))
}

