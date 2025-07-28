rm(list = ls())

library(httr)
library(jsonlite)

external_study_url <- "https://kyblab.tuebingen.mpg.de/mex/wm-tasks/experiments/symmetry_span_task.html?PROLIFIC_PID={{%PROLIFIC_PID%}}"
# 🔐 Your API token
api_token <- c(
  "REMOVED_API_TOKEN"
)
base_url <- "https://api.prolific.com/api/v1"

workspace_response <- httr::GET(
  url = "https://api.prolific.com/api/v1/workspaces/",
  httr::add_headers(
    Authorization = paste("Token", api_token),
    "Content-Type" = "application/json"
  )
)

workspace_data <- httr::content(workspace_response, as = "parsed")
workspace_id <- workspace_data$results[[2]]$id  # Adjust if you have multiple workspaces

headers <- add_headers(
  Authorization = paste("Token", api_token),
  "Content-Type" = "application/json"
)


# create the study --------------------------------------------------------



study_payload <- jsonlite::toJSON(list(
  name = "Pick the Oddball!",
  internal_name = "triplet ooo ID: demographics, personality, & psychiatry",
  description = paste(
    "Your main task is to identify the odd one out from a set of three simultaneously presented objects.",
    "This is a lengthy and cognitively demanding task—comprising 440 odd-one-out trials in total.",
    "Before you begin, you'll need to read the instructions carefully and pass a comprehension check.",
    "Once you've completed the main task, you'll be asked to provide demographic details and complete several questionnaires.",
    "The entire study will take approximately 75 minutes. You’ll receive a reward of £10 upon successful completion.",
    "Payments are typically processed within 48 hours.",
    sep = "\n"
  ),
  study_type = "QUOTA",
  external_study_url = "https://kyblab.tuebingen.mpg.de/mex/wm-tasks/experiments/symmetry_span_task.html?PROLIFIC_PID={{%PROLIFIC_PID%}}",
  prolific_id_option = "url_parameters",
  completion_option = "url",
  completion_code = "UEIONKBASDOGK837955K8395",  # simplified
  total_available_places = 250,
  estimated_completion_time = 75,
  maximum_allowed_time = 164,
  reward = 1000,  # in pence
  device_compatibility = list("desktop"),
  workspace_id = workspace_id
), auto_unbox = TRUE)


study_response <- POST(
  url = paste0(base_url, "/studies/"),
  body = study_payload,
  headers
)

study <- content(study_response, as = "parsed")
print(paste("Study created with ID:", study$id))
print(paste("workspace ID:", study$workspace))
# workspace should have been added, but still in "My Workspace"...
# workspace id has to be added again via PATCH, is not added otherwise for some reason
workspace_response <- PATCH(
  url = paste0(base_url, "/studies/", study$id),
  body = jsonlite::toJSON(list(workspace_id = workspace_id), auto_unbox = TRUE),
  headers
)
ws <- content(workspace_response)

print(paste("workspace ID:", ws$workspace))


# only then attach the age quota filter -----------------------------------
filter_payload <- list(
  workspace_id = workspace_id,
  filters = list(
    list(
      filter_id = "age",
      selected_range = list(lower = 18, upper = 70),
      weightings = list(
        "group_1" = list(
          selected_range = list(lower = 18, upper = 20),
          weighting = 1
        ),
        "group_2" = list(
          selected_range = list(lower = 21, upper = 30),
          weighting = 1
        ), 
        "group_3" = list(
          selected_range = list(lower = 31, upper = 60),
          weighting = 1
        ), 
        "group_4" = list(
          selected_range = list(lower = 61, upper = 70),
          weighting = 1
        )
      )
    )
  )
)


filter_json <- toJSON(filter_payload, auto_unbox = TRUE, pretty = TRUE)

patch_response <- PATCH(
  url = paste0("https://api.prolific.com/api/v1/studies/", study$id, "/"),
  body = filter_json,
  encode = "json",
  headers
)
content(patch_response, as = "parsed")
