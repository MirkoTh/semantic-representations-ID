// get subject ID
function getQueryVariable(variable) {
  var query = window.location.search.substring(1);
  var vars = query.split("&");
  for (var i = 0; i < vars.length; i++) {
    var pair = vars[i].split("=");
    if (pair[0] == variable) {
      return pair[1];
    }
  }
  return false;
}

function download(content, fileName, contentType) {
  var a = document.createElement("a");
  var file = new Blob([content], { type: contentType });
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
}

function prepare_recall(data_recall) {
  var trial_id_recall = data_recall.select("trial_id_recall");
  var set_size = data_recall.select("set_size");
  var stimuli = data_recall.select("stimuli");
  var responses = data_recall.select("recall");
  var n_correct = data_recall.select("accuracy");
  var rt = data_recall.select("rt");
  var data_recall_clean = {
    participant_id: participant_id,
    trial_id_recall: trial_id_recall,
    set_size: set_size,
    stimuli: stimuli,
    response: responses,
    n_correct: n_correct,
    rt: rt,
  };
  return data_recall_clean;
}

function prepare_processing(data_processing) {
  var trial_id_recall = data_processing.select("trial_id_recall");
  var trial_id_processing = data_processing.select("trial_id_processing");
  var set_size = data_processing.select("set_size");
  var accuracy = data_processing.select("accuracy");
  var rt = data_processing.select("rt");
  var data_clean = {
    trial_id_recall: trial_id_recall,
    trial_id_processing: trial_id_processing,
    set_size: set_size,
    accuracy: accuracy,
    rt: rt,
  };
  return data_clean;
}

function make_rect(x_start, y_start, stepsize_x, stepsize_y) {
  const rect_object = {
    obj_type: "rect",
    startX: screen.width / 2 + x_start, //x_start, // location of the rectangle's center in the canvas
    startY: screen.height / 2 + y_start,
    width: stepsize_x,
    height: stepsize_y,
    line_color: "black", // You can use the HTML color name instead of the HEX color.
    line_width: 3,
    show_start_time: 0, // ms after the start of the trial
  };
  return rect_object;
}

function saveData(filedata, filename, task, flag_save = true) {
  //var filename = "./data/" + task + "-participant-" + participant_id + ".json";
  var filename_folder = "../.././data/" + task + "/" + filename;
  $.post("save_data.php", {
    postresult: filedata + "\n",
    postfile: filename_folder,
  });
  if (flag_save) {
    dataSaved = true;
  }
}

function checkDataSaving() {
  if (!dataSaved) {
    // Data saving is not complete, wait for a few seconds and check again.
    setTimeout(checkDataSaving, 2000); // Wait for 2 seconds and check again.
  } else {
    // Data saving is complete, you can proceed with the next steps.
    console.log("Data has been saved successfully.");
  }
}

async function saveSeveralData(filedata, filename, task) {
  //var filename = "./data/" + task + "-participant-" + participant_id + ".json";
  var filename_folder = "../.././data/" + task + "/" + filename;
  var n_data = filedata.length;
  for (var i = 0; i < n_data; i++) {
    $.post("save_data.php", {
      postresult: JSON.stringify(filedata[i]) + "\n",
      postfile: filename_folder,
    });
  }
  dataSaved = true;
}

async function saveSeveralDataOverwrite(filedata, filename, task) {
  //var filename = "./data/" + task + "-participant-" + participant_id + ".json";
  var filename_folder = "../.././data/" + task + "/" + filename;
  var n_data = filedata.length;
  for (var i = 0; i < n_data; i++) {
    if (i == 0) {
      $.post("save_data_overwrite.php", {
        postresult: JSON.stringify(filedata[i]) + "\n",
        postfile: filename_folder,
      });
    } else if (i > 0) {
      $.post("save_data.php", {
        postresult: JSON.stringify(filedata[i]) + "\n",
        postfile: filename_folder,
      });
    }
  }
  dataSaved = true;
}

function updateQueryStringParameter(uri, key, value) {
  var re = new RegExp("([?&])" + key + "=.*?(&|$)", "i");
  var separator = uri.indexOf("?") !== -1 ? "&" : "?";
  if (uri.match(re)) {
    return uri.replace(re, "$1" + key + "=" + value + "$2");
  } else {
    return uri + separator + key + "=" + value;
  }
}

// UPDATING

var trial_type_updating = [
  [
    "update",
    "immediate",
    "immediate",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "immediate",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "immediate",
    "update",
    "immediate",
    "update",
    "update",
    "immediate",
    "update",
  ],
  [
    "update",
    "immediate",
    "update",
    "update",
    "immediate",
    "update",
    "immediate",
    "immediate",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "update",
    "immediate",
    "immediate",
    "update",
    "update",
  ],
];

var immediate_set_updating = [
  [
    [4, 6, 2, 0, 3],
    [6, 0, 3, 2, 5],
    [9, 8, 4, 7, 0],
    [9, 1, 2, 0, 5],
    [5, 2, 6, 8, 7],
    [0, 4, 8, 3, 5],
  ],
  [
    [8, 3, 0, 5, 2],
    [2, 9, 4, 1, 8],
    [4, 2, 0, 8, 7],
    [4, 2, 8, 9, 3],
    [9, 3, 2, 6, 7],
    [8, 6, 5, 7, 2],
  ],
];

var initial_set_updating = [
  [
    [9, 6, 7, 0, 2],
    [3, 6, 9, 7, 0],
    [9, 5, 2, 3, 4],
    [2, 8, 3, 0, 5],
    [3, 1, 7, 5, 6],
    [9, 8, 4, 6, 0],
    [9, 2, 7, 1, 4],
    [8, 2, 6, 1, 4],
    [4, 0, 9, 8, 7],
    [9, 3, 7, 5, 6],
    [2, 4, 1, 3, 5],
    [5, 7, 2, 8, 6],
    [9, 1, 5, 7, 6],
    [6, 4, 5, 1, 0],
    [2, 7, 5, 1, 9],
    [3, 9, 8, 0, 1],
    [1, 4, 9, 8, 2],
    [9, 8, 1, 3, 6],
    [1, 3, 6, 5, 9],
    [7, 1, 8, 2, 9],
    [1, 5, 4, 2, 6],
  ],
  [
    [5, 4, 1, 0, 9],
    [6, 3, 0, 5, 2],
    [4, 1, 8, 3, 2],
    [0, 2, 1, 8, 4],
    [0, 4, 9, 3, 5],
    [2, 1, 9, 0, 3],
    [5, 8, 3, 4, 0],
    [3, 1, 4, 2, 0],
    [3, 7, 8, 1, 4],
    [5, 0, 4, 1, 9],
    [6, 8, 0, 9, 3],
    [5, 2, 9, 0, 3],
    [5, 6, 2, 8, 9],
    [4, 6, 5, 9, 8],
    [1, 2, 9, 4, 0],
    [3, 0, 2, 5, 6],
    [6, 9, 3, 5, 7],
    [9, 7, 1, 3, 5],
    [1, 7, 6, 3, 4],
    [7, 2, 3, 0, 4],
    [8, 3, 0, 5, 1],
  ],
];

var locations_update_updating = [
  [
    [4, 0, 1, 2, 0, 0, 2],
    [0, 3, 2, 1, 4, 0, 2],
    [2, 4, 0, 0, 0, 2, 1],
    [2, 4, 3, 2, 1, 3, 0],
    [4, 3, 4, 0, 2, 1, 2],
    [3, 2, 0, 0, 0, 0, 4],
    [3, 0, 1, 0, 1, 4, 4],
    [1, 1, 4, 0, 0, 1, 0],
    [3, 1, 1, 2, 4, 2, 3],
    [1, 1, 2, 2, 2, 4, 0],
    [1, 0, 0, 3, 3, 1, 3],
    [0, 4, 0, 2, 0, 1, 3],
    [2, 1, 2, 4, 0, 4, 3],
    [1, 4, 0, 1, 3, 4, 1],
    [4, 0, 2, 3, 2, 1, 2],
    [1, 2, 0, 4, 1, 4, 2],
    [4, 3, 1, 3, 3, 0, 3],
    [4, 3, 0, 3, 0, 1, 1],
    [4, 2, 1, 4, 1, 2, 2],
    [4, 2, 3, 1, 2, 4, 0],
    [3, 1, 4, 0, 2, 4, 4],
  ],
  [
    [1, 4, 4, 0, 1, 2, 2],
    [0, 1, 4, 2, 2, 2, 2],
    [0, 2, 2, 2, 4, 3, 3],
    [0, 0, 0, 0, 1, 0, 0],
    [4, 0, 4, 1, 2, 1, 1],
    [3, 0, 1, 2, 3, 4, 3],
    [0, 2, 1, 3, 4, 4, 3],
    [0, 1, 4, 3, 3, 4, 3],
    [2, 3, 1, 1, 4, 0, 2],
    [0, 0, 4, 0, 3, 0, 1],
    [2, 0, 3, 0, 3, 4, 2],
    [4, 0, 1, 1, 4, 0, 2],
    [2, 1, 0, 3, 1, 1, 1],
    [2, 2, 4, 0, 2, 4, 4],
    [2, 2, 4, 1, 1, 2, 4],
    [2, 2, 0, 0, 2, 4, 4],
    [3, 4, 3, 2, 1, 0, 0],
    [0, 4, 1, 1, 2, 1, 0],
    [0, 1, 3, 0, 4, 3, 4],
    [4, 1, 2, 1, 1, 0, 0],
    [3, 0, 1, 1, 1, 0, 3],
  ],
];

var items_replace_updating = [
  [
    [0, 1, 8, 9, 6, 4, 7],
    [0, 2, 6, 1, 5, 2, 7],
    [3, 1, 7, 2, 0, 3, 1],
    [2, 4, 9, 8, 6, 9, 0],
    [3, 4, 4, 3, 0, 1, 5],
    [2, 1, 4, 9, 2, 8, 9],
    [4, 9, 8, 3, 4, 6, 7],
    [6, 9, 4, 6, 0, 0, 2],
    [7, 2, 5, 9, 5, 1, 0],
    [9, 9, 4, 9, 8, 9, 5],
    [5, 9, 1, 9, 7, 4, 8],
    [5, 4, 4, 6, 4, 2, 7],
    [4, 1, 0, 1, 9, 1, 8],
    [0, 9, 8, 4, 9, 9, 6],
    [8, 1, 5, 9, 3, 7, 2],
    [6, 3, 8, 3, 6, 8, 1],
    [0, 2, 1, 5, 8, 8, 2],
    [5, 4, 4, 7, 9, 3, 2],
    [4, 1, 4, 6, 9, 9, 0],
    [2, 8, 1, 3, 6, 9, 8],
    [1, 2, 9, 1, 6, 1, 1],
  ],
  [
    [6, 3, 9, 0, 2, 6, 6],
    [7, 5, 8, 5, 5, 2, 2],
    [1, 8, 6, 0, 4, 3, 8],
    [0, 2, 6, 2, 3, 3, 7],
    [1, 4, 8, 8, 2, 1, 9],
    [2, 7, 4, 3, 9, 7, 6],
    [5, 1, 0, 5, 8, 5, 1],
    [0, 3, 6, 6, 0, 6, 0],
    [8, 3, 9, 3, 9, 3, 5],
    [8, 3, 7, 1, 4, 0, 5],
    [9, 4, 7, 9, 2, 7, 5],
    [8, 3, 5, 0, 0, 2, 7],
    [3, 4, 6, 6, 9, 5, 1],
    [7, 7, 6, 2, 3, 4, 5],
    [4, 9, 4, 3, 4, 1, 0],
    [5, 4, 2, 4, 5, 0, 0],
    [5, 2, 2, 7, 7, 2, 9],
    [2, 7, 1, 2, 0, 0, 8],
    [8, 8, 4, 5, 4, 4, 6],
    [1, 4, 4, 0, 3, 1, 6],
    [3, 0, 0, 6, 2, 6, 5],
  ],
];

var final_set_updating = [
  [
    [4, 8, 7, 0, 0],
    [2, 1, 7, 2, 5],
    [0, 1, 3, 3, 1],
    [0, 6, 8, 9, 4],
    [3, 1, 5, 4, 4],
    [8, 8, 1, 2, 9],
    [3, 4, 7, 4, 7],
    [2, 0, 6, 1, 4],
    [4, 5, 1, 0, 5],
    [5, 9, 8, 5, 9],
    [1, 4, 1, 8, 5],
    [4, 2, 6, 7, 4],
    [9, 1, 0, 8, 1],
    [8, 6, 5, 9, 9],
    [1, 7, 2, 9, 8],
    [8, 6, 1, 0, 8],
    [8, 1, 9, 2, 0],
    [9, 2, 1, 7, 5],
    [1, 9, 0, 5, 6],
    [8, 3, 6, 1, 9],
    [1, 2, 6, 1, 1],
  ],
  [
    [0, 2, 6, 0, 9],
    [7, 5, 2, 5, 8],
    [1, 1, 0, 8, 4],
    [7, 3, 1, 8, 4],
    [4, 9, 2, 3, 8],
    [7, 4, 3, 6, 7],
    [5, 0, 1, 1, 5],
    [0, 3, 4, 0, 6],
    [3, 3, 5, 3, 9],
    [0, 5, 4, 4, 7],
    [9, 8, 5, 2, 7],
    [2, 0, 7, 0, 0],
    [6, 1, 3, 6, 9],
    [2, 6, 3, 9, 5],
    [1, 4, 1, 4, 0],
    [4, 0, 5, 5, 0],
    [9, 7, 7, 2, 2],
    [8, 0, 0, 3, 7],
    [5, 8, 6, 4, 6],
    [6, 3, 4, 0, 1],
    [6, 2, 0, 5, 1],
  ],
];

// COMPREHENSION CHECK OOO

       // how many images are going to be displayed?
        // what is your task when the three objects are displayed on screen?
        // how do you select the image you find the most different from the others?
        // how do you respond the questions?
        // how do you redirect to prolific to confirm the successful completion of the study?


var comprehension_question_ooo_allinone = {
  type: jsPsychSurveyMultiChoice,
  questions: [
    {
      prompt:
        "<div align=center><b>How many images are going to be displayed in every trial?</b></div>",
      options: [
        "Sometimes two, but sometimes three, or four.",
        "It is a pairwise similarity judgment task, so two items are displayed in every trial.",
        "It is a triplet similarity judgment task, so three items are displayed in every trial.",
        "The number of items is randomly determined by a human-like large language model.",
      ],
      required: true,
    },
    {
      prompt: "<div align=center><b>What is your task when the three objects are displayed on screen?</b></div>",
      options: [
        "Click on the most similar item with the mouse.",
        "Click on the item the least similar to the remaining two items.",
        "Report the pairwise similarity between all three possible pairs.",
        "Rate all three objects sequentially.",
      ],
      required: true,
    },
    {
      prompt:
        "<div align=center><b>How do you select the image you find the most different from the others?</b></div>",
      options: [
        "Select it with a mouse click.",
        "Select it by pressing L (left), M (middle), or R (right) on the keyboard.",
        "Unselect the maximally similar items with double mouse clicks.",
        "Select it via the speech recognition system.",
      ],

      required: true,
    },
    
  ],
  preamble: "<h3>Please answer the following question.</h3>",
  randomize_question_order: true,

  on_finish: function (data) {
    //var data = jsPsych.data.getLastTrialData().values()[0];
    var answer_Q1 = data.response.Q0;
    var answer_Q2 = data.response.Q1;
    var answer_Q3 = data.response.Q2;

    if (
      answer_Q1 ==
        "It is a triplet similarity judgment task, so three items are displayed in every trial." &&
      answer_Q2 ==
        "Click on the item the least similar to the remaining two items." &&
      answer_Q3 ==
        "Select it with a mouse click.") {
      data.correct = true;
    } else {
      data.correct = false;
    }
  },
};

//compcheck1: this function returns feedback based on response given
var comp_feedback = {
  type: jsPsychHtmlButtonResponse,
  stimulus: function () {
    var last_resp_correct = jsPsych.data.get().last(1).filter({ correct: true });
    if (last_resp_correct.count() == 1) {
      return "<p align='center'><b>Well done! You answered all questions correctly.</b></p>";
    } else {
      return "<p align='center'><b>Not all questions were answered correctyly.</b> Please try again. </p>";
    }
  },
  choices: ["Next"],
};

var comp_feedback_ooo_verbose = {
  type: jsPsychHtmlButtonResponse,
  stimulus: function () {
    var last_resp_correct = jsPsych.data.get().last(1).filter({ correct: true });
    if (last_resp_correct.count() == 1) {
      var info =
        "<p align='center'><b>Well done! You answered all questions correctly.<br><br></b></p>";
    } else {
      var info =
        "<p align='center'><b>Not all questions were answered correctly.<br>Below you get feedback, which questions you got wrong.<br>Correct responses are printed in <span style='color: green'>green</span>, incorrect responses in <span style='color: red'>red</span>.<br><br></b> Please try again. <br><br></p>";
    }

    var q_responses = jsPsych.data.get().last(1).values();
    var answer_Q1 = q_responses[0].response.Q0;
    var answer_Q2 = q_responses[0].response.Q1;
    var answer_Q3 = q_responses[0].response.Q2;
    var qna1 =
      "<b>How many images are going to be displayed in every trial?<br>Your response: </b>" +
      answer_Q1 +
      "<br>";
    var qna2 =
      "<b>What is your task when the three objects are displayed on screen?<br>Your response: </b>" +
      answer_Q2 +
      "<br>";
    var qna3 =
      "<b>How do you select the image you find the most different from the others?<br>Your response: </b>" +
      answer_Q3 +
      "<br>";


    var explain1 =
      "<b>Hint:</b> It is a triplet similarity judgment task.<br><br></p>";
    var explain2 =
      "<b>Hint:</b> Find the outlier, i.e., the object that is most different from the other two.<br><br></p>";
    var explain3 =
      "<b>Hint:</b> Just use your mouse for that.<br><br></p>";

    if (
      answer_Q1 == 
      "It is a triplet similarity judgment task, so three items are displayed in every trial.") {
      var t1 =
        '<p style="color:green;align=center">' + "<b>CORRECT! </b>" + qna1;
    } else {
      var t1 =
        '<p style="color:red;align=center"">' +
        "<b>INCORRECT! </b>" +
        qna1 +
        explain1;
    }
    if (
      answer_Q2 ==
      "Click on the item the least similar to the remaining two items."
    ) {
      var t2 =
        '<p style="color:green;align=center"">' + "<b>CORRECT! </b>" + qna2;
    } else {
      var t2 =
        '<p style="color:red;align=center"">' +
        "<b>INCORRECT! </b>" +
        qna2 +
        explain2;
    }
    if (
      answer_Q3 ==
      "Select it with a mouse click."
    ) {
      var t3 =
        '<p style="color:green;align=center"">' + "<b>CORRECT! </b>" + qna3;
    } else {
      var t3 =
        '<p style="color:red;align=center"">' +
        "<b>INCORRECT! </b>" +
        qna3 +
        explain3;
    }
    var pg = info + t1 + t2 + t3;

    return pg;
  },
  choices: ["Next"],
};

//function to hide one html div and show another
function clickStart(hide, show) {
  document.getElementById(hide).style.display = "none";
  document.getElementById(show).style.display = "block";
  window.scrollTo(0, 0);
}

//function to hide one html div and show another
function myDataProtection() {
  document.getElementById(
    "DataProtection"
  ).innerHTML = `The processing and use of the collected data occurs in a pseudoanonymized form within the scope of the legally prescribed provisions.

As a general rule, the storage occurs in the form of answered questionnaires, as well as electronic data, for a duration of 10 years or longer, if this is required by the purpose of the study.
    
By providing of further personal data in pseudoanonymized form, collected personal data may be used for the preparation of anonymized
scientific research work and may also be published and used in an anonymized form in medical journals and scientific publications, so that a direct assignment to my person cannot be established.

The information obtained during the course of this study may also be sent in an anonymized form to cooperation partners within the scope of
the European General Data Protection Regulation for scientific purposes and to cooperation partners outside of the European Union, i.e. to countries with a lower data protection level (this also applies to the USA).

The data collected within the scope of the study can also be used and processed in the future inside of the Max Planck Institute.

I was informed about my rights, that at any time:

I can withdraw this declaration of consent.

I can request information about my stored data and request the correction or blocking of data.

By cancellation of my participation in the study,
I can request that any personal data of mine collected until that point are immediately deleted or anonymized.

I can request that my personal data are handed out to me or to third parties (if technically feasible).

I herewith declare that:

I have been adequately informed about the collection and processing of my personal data and rights.

I consent to the collection and processing of personal data within the scope of the study and its pseudoanonymized disclosure,
so that only the persons conducting the study can establish a link between the data and my person.

Agreement:

I agree to participate in this study.

I consent to the use of my data described in the Data Protection Information Sheet
and confirm having received a copy of the Data Protection Sheet.

I consent to data transfer from the MPI for Biological Cybernetics encrypted database to the project-related collaborators:
inside of the Max Planck Society and affiliated research institutes, or at partnering institutions like the University of Tuebingen.`;
}

function direct_to_ooo() {


  if (window.location.search.indexOf("PROLIFIC_PID") > -1) {
    var participant_id = getQueryVariable("PROLIFIC_PID");
  }

  // If no ID is present, generate one using random numbers - this is useful for testing
  else {
    var participant_id = Math.floor(Math.random() * 1000);
  }

  if (window.location.search.indexOf("SESSION_ID") > -1) {
    var session_id = getQueryVariable("SESSION_ID");
  }

  else {
    // only one session anyway
    var session_id = 1;
  }

 

  jatos.setStudySessionData({"participant_id": participant_id, "session_id":session_id})

  jatos.startComponent(66)
}

function direct_to_qs() {
  if (window.location.search.indexOf("PROLIFIC_PID") > -1) {
    var participant_id = getQueryVariable("PROLIFIC_PID");
  }
  // If no ID is present, generate one using random numbers - this is useful for testing
  else {
    var participant_id = Math.floor(Math.random() * 1000);
  }

  window.location.href = progress_url;
}

var instructions0 = {
  type: jsPsychSurveyMultiChoice,
  questions: [
    {
      prompt: `<div style="font-size:30px;"><b>IMPORTANT</b><br><br><br>
        Your data are used for scientific purposes.<br>
        We have invested a lot of time to develop this study.<br>
        <u>Please answer to the best of your knowledge and belief; otherwise, we cannot use your data at all!</u><br>
        Please commit to being honest.`,
      options: ["I commit to being honest"],

      required: true,
      name: "question1",
    },
  ],
  preamble:
    "<div style='font-size:40px;color:red'><u>ANSWER TO THE BEST OF YOUR KNOWLEDGE AND BELIEF!</u><br><br></div>",
};
