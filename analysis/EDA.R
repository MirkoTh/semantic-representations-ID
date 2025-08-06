library(jatosR)
library(tidyverse)
library(jsonlite)
library(httr2)

library(jsonlite)

# Read the file as text
raw <- readLines("data/study_1/study_result_253/comp-result_304/files/odd-one-out.csv")


# Collapse lines into single string
json_str <- paste(raw, collapse = "\n")

# Parse to a list
parsed <- fromJSON(json_str)

str(parsed)
tbl_ooo <- as_tibble(parsed)
