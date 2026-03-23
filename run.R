library(shiny)
library(shinyjs)
library(jsonlite)
library(shinyBS)

get_available_tasks <- function(python_bin = "python") {
  cmd <- sprintf("%s list_tasks.py", shQuote(python_bin))
  out <- tryCatch(
    system(cmd, intern = TRUE),
    error = function(e) NULL
  )
  if (is.null(out) || length(out) == 0) return(NULL)
  tryCatch(fromJSON(paste(out, collapse = "\n")), error = function(e) NULL)
}


get_available_horizons <- function(labels_dir = file.path("synthetic_amr_graphs_train", "labels")) {
  manifest_path <- file.path(labels_dir, "manifest.json")
  if (!file.exists(manifest_path)) return(NULL)
  obj <- tryCatch(fromJSON(manifest_path), error = function(e) NULL)
  if (is.null(obj) || is.null(obj$horizons)) return(NULL)
  hs <- suppressWarnings(as.integer(obj$horizons))
  hs <- hs[!is.na(hs)]
  if (length(hs) == 0) return(NULL)
  sort(unique(hs))
}

ui <- fluidPage(
  useShinyjs(),
  
  titlePanel("Digital Twins (AMR)"),
  sidebarLayout(
    sidebarPanel(
      h4("Data & Task"),
      
      hr(),
      checkboxInput(
        "use_task_hparams",
        "Let task override hparams (--use_task_hparams)",
        value = TRUE
      ),
      
      checkboxInput(
        "train_model",
        "Train model (if unchecked: use saved model only)",
        value = TRUE
      ),
      
      br(),
      actionButton("run_model", "Run model", class = "btn btn-primary"),
      
      br(),
      helpText("Training data folder is fixed to: synthetic_amr_graphs_train"),
      helpText("Test data folder is fixed to: synthetic_amr_graphs_test"),
      
      br(),
      
      selectInput(
        "task",
        "Task",
        choices = c("Loading tasks…" = ""),
        selected = ""
      ),
      
      # NEW: Horizon control (applies only to tasks that end in _h<digits>)
      numericInput(
        "pred_horizon",
        "Prediction horizon H (days)",
        value = 7,
        min = 1,
        step = 1
      ),
      
      numericInput("T", "Window length T", value = 7, min = 1, step = 1),
      
      div(
        id = "hparam_panel",
        
        sliderInput(
          "sliding_step",
          "Sliding step",
          min = 1,
          max = 10,
          value = 1,
          step = 1
        ),
        
        hr(),
        h4("Model hyperparameters"),
        
        sliderInput(
          "hidden",
          "Hidden size",
          min = 16,
          max = 256,
          value = 64,
          step = 16
        ),
        
        numericInput("heads", "Transformer heads", value = 2, min = 1, step = 1),
        sliderInput("dropout", "Dropout", min = 0, max = 0.9, value = 0.2, step = 0.05),
        
        sliderInput(
          "transformer_layers",
          "Number of Transformer layers",
          min = 1,
          max = 10,
          value = 2,
          step = 1
        ),
        
        sliderInput(
          "sage_layers",
          "Number of GraphSAGE layers (depth)",
          min = 1,
          max = 10,
          value = 3,
          step = 1
        ),
        
        checkboxInput("use_cls", "Use [CLS] token", value = FALSE),
        
        hr(),
        h4("Training hyperparameters"),
        
        sliderInput(
          "batch_size",
          "Batch size",
          min = 8,
          max = 256,
          value = 16,
          step = 8
        ),
        
        numericInput("epochs", "Epochs", value = 20, min = 1, step = 1),
        numericInput("lr", "Learning rate", value = 1e-4, min = 1e-6, step = 1e-5),
        
        hr(),
        numericInput(
          "max_neighbors",
          "Legacy max_neighbors per node (edge-thinning; ignored if neighbor_sampling=TRUE)",
          value = 20,
          min = 0,
          step = 1
        )
      ),
      
      hr(),
      h4("Graph sampling (always user-controlled)"),
      
      checkboxInput(
        "neighbor_sampling",
        "Use true GraphSAGE neighbor sampling (--neighbor_sampling)",
        value = FALSE
      ),
      
      textInput(
        "num_neighbors",
        "num_neighbors per hop (comma-separated, e.g. 15,10)",
        value = "15,10"
      ),
      
      numericInput(
        "seed_count",
        "seed_count (number of seed nodes per graph)",
        value = 256,
        min = 0,
        step = 1
      ),
      
      selectInput(
        "seed_strategy",
        "seed_strategy",
        choices = c("random", "all"),
        selected = "random"
      ),
      
      numericInput(
        "seed_batch_size",
        "seed_batch_size (seeds per sampled subgraph batch)",
        value = 64,
        min = 1,
        step = 1
      ),
      
      numericInput(
        "max_sub_batches",
        "max_sub_batches (cap sampled subgraph batches per graph; 0 = no cap)",
        value = 4,
        min = 0,
        step = 1
      ),
      
      hr(),
      h4("Attention heatmap"),
      sliderInput(
        "attn_top_k",
        "Top-K nodes to display",
        min = 10,
        max = 200,
        value = 10,
        step = 5
      ),
      selectInput(
        "attn_rank_by",
        "Rank nodes by",
        choices = c("abs_diff", "mean"),
        selected = "abs_diff"
      ),
      
      hr(),
      textInput(
        "python_bin",
        "Python executable",
        value = "python"
      ),
      
      # ------------------------------------------------------------
      # Tooltips (shinyBS)
      # ------------------------------------------------------------
      bsTooltip("use_task_hparams", "If checked, the selected task overrides the model/training hyperparameters.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("train_model", "If unchecked, skips training and only runs evaluation using a saved model (if present).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("run_model", "Runs train_amr_dygformer.py with the arguments shown in Command preview.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("task", "Select the prediction task (classification or regression) as defined in tasks.py.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("pred_horizon", "Prediction horizon (days ahead). If the chosen task name ends with _h<digits>, the UI swaps it to _hH before calling Python. If the task has no _h suffix, this is ignored.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("T", "Temporal window length (number of days/time steps per input sequence).", placement = "right", trigger = "hover", options = list(container = "body")),
      
      bsTooltip("sliding_step", "Stride used to slide the T-length window through time (smaller = more overlap, more samples).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("hidden", "Transformer hidden dimension (embedding size). Larger can improve capacity but increases compute.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("heads", "Number of attention heads in the Transformer.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("dropout", "Dropout probability for regularisation (higher can reduce overfitting but may underfit).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("transformer_layers", "Number of stacked Transformer layers (depth of temporal modelling).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("sage_layers", "Number of GraphSAGE layers (depth of spatial message passing).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("use_cls", "If checked, uses a [CLS] token style pooling for sequence-level prediction (when supported).", placement = "right", trigger = "hover", options = list(container = "body")),
      
      bsTooltip("batch_size", "Training batch size (higher uses more memory; may speed up training).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("epochs", "Number of training epochs.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("lr", "Learning rate for optimiser (too high may diverge; too low may be slow).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("max_neighbors", "Legacy edge-thinning cap per node. Disabled automatically when neighbor_sampling=TRUE.", placement = "right", trigger = "hover", options = list(container = "body")),
      
      bsTooltip("neighbor_sampling", "Enable true GraphSAGE neighbor sampling during training/eval.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("num_neighbors", "Comma-separated neighbor counts per hop (e.g., '15,10' for 2 hops).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("seed_count", "Number of seed nodes sampled per graph when neighbor sampling is enabled.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("seed_strategy", "How to choose seed nodes: random subset or all nodes.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("seed_batch_size", "Seeds per sampled subgraph batch (controls number/size of sampled subgraphs).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("max_sub_batches", "Maximum sampled subgraph batches per graph (0 means no cap).", placement = "right", trigger = "hover", options = list(container = "body")),
      
      bsTooltip("attn_top_k", "Number of top-ranked nodes to show in attention heatmap outputs.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("attn_rank_by", "Ranking statistic for attention nodes (abs_diff vs mean).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("python_bin", "Python executable used to run list_tasks.py and train_amr_dygformer.py.", placement = "right", trigger = "hover", options = list(container = "body"))
    ),
    
    mainPanel(
      h4("Command preview"),
      verbatimTextOutput("cmd_preview"),
      
      h4("Run log"),
      verbatimTextOutput("run_log"),
      
      tabsetPanel(
        tabPanel(
          "Loss curves",
          br(),
          imageOutput("loss_curves_img", height = "100%")
        ),
        tabPanel(
          "Confusion matrix (val)",
          br(),
          imageOutput("cm_img", height = "100%")
        ),
        tabPanel(
          "ROC curve (val)",
          br(),
          imageOutput("roc_img", height = "100%")
        ),
        tabPanel(
          "Test confusion matrix",
          br(),
          imageOutput("cm_test_img", height = "100%")
        ),
        tabPanel(
          "Test ROC curve",
          br(),
          imageOutput("roc_test_img", height = "100%")
        ),
        tabPanel(
          "Attention heatmap (val)",
          br(),
          imageOutput("attn_heatmap_img", height = "100%")
        ),
        tabPanel(
          "Attention heatmap (test)",
          br(),
          imageOutput("attn_heatmap_test_img", height = "100%")
        )
      )
    )
  )
)

server <- function(input, output, session) {

  available_horizons <- reactive({
    get_available_horizons()
  })

  observe({
    hs <- available_horizons()
    if (!is.null(hs)) {
      Hmin <- min(hs)
      Hmax <- max(hs)
      Hcur <- as.integer(input$pred_horizon)
      if (is.na(Hcur)) Hcur <- Hmin
      if (Hcur < Hmin) Hcur <- Hmin
      if (Hcur > Hmax) Hcur <- Hmax
      updateNumericInput(session, "pred_horizon", min = Hmin, max = Hmax, value = Hcur)
    }
  })

  
  # Disable manual hparam panel when task overrides are enabled
  observe({
    if (isTRUE(input$use_task_hparams)) {
      shinyjs::disable("hparam_panel")
    } else {
      shinyjs::enable("hparam_panel")
    }
  })
  
  # Disable max_neighbors when neighbor_sampling is enabled
  observe({
    if (isTRUE(input$neighbor_sampling)) {
      shinyjs::disable("max_neighbors")
    } else {
      shinyjs::enable("max_neighbors")
    }
  })
  
  # NEW: Disable horizon input when selected task has no _h<digits> suffix
  observe({
    task_id <- as.character(input$task)
    has_h <- grepl("_h[0-9]+$", task_id)
    if (isTRUE(has_h)) {
      shinyjs::enable("pred_horizon")
    } else {
      shinyjs::disable("pred_horizon")
    }
  })
  
  log_text <- reactiveVal("")
  output$run_log <- renderText({ log_text() })
  
  run_dir <- reactiveVal("training_outputs")
  
  # NEW: resolve the effective task name (swap trailing horizon if present)
  effective_task <- reactive({
    task_id <- as.character(input$task)
    if (is.na(task_id) || task_id == "") return("")
    if (grepl("_h[0-9]+$", task_id)) {
      H <- as.integer(input$pred_horizon)
      if (is.na(H) || H < 1) H <- 1
      return(sub("_h[0-9]+$", paste0("_h", H), task_id))
    }
    task_id
  })
  
  build_args <- reactive({
    data_folder <- "synthetic_amr_graphs_train"
    
    args <- c(
      "train_amr_dygformer.py",
      "--data_folder", data_folder,
      "--task", effective_task(),
      "--T", as.character(input$T)
    )
    
    if (isTRUE(input$use_task_hparams)) {
      args <- c(args, "--use_task_hparams")
    } else {
      args <- c(
        args,
        "--sliding_step", as.character(input$sliding_step),
        "--hidden", as.character(input$hidden),
        "--heads", as.character(input$heads),
        "--dropout", as.character(input$dropout),
        "--transformer_layers", as.character(input$transformer_layers),
        "--sage_layers", as.character(input$sage_layers),
        "--batch_size", as.character(input$batch_size),
        "--epochs", as.character(input$epochs),
        "--lr", as.character(input$lr)
      )
      
      if (isTRUE(input$use_cls)) {
        args <- c(args, "--use_cls")
      }
      
      args <- c(args, "--max_neighbors", as.character(input$max_neighbors))
    }
    
    ns_flag <- if (isTRUE(input$neighbor_sampling)) "true" else "false"
    args <- c(args, "--neighbor_sampling", ns_flag)
    
    if (isTRUE(input$neighbor_sampling)) {
      args <- c(
        args,
        "--num_neighbors", as.character(input$num_neighbors),
        "--seed_count", as.character(input$seed_count),
        "--seed_strategy", as.character(input$seed_strategy),
        "--seed_batch_size", as.character(input$seed_batch_size),
        "--max_sub_batches", as.character(input$max_sub_batches),
        "--max_neighbors", "0"
      )
    }
    
    # Attention heatmap controls
    args <- c(
      args,
      "--attn_top_k", as.character(input$attn_top_k),
      "--attn_rank_by", as.character(input$attn_rank_by)
    )
    
    train_flag <- if (isTRUE(input$train_model)) "true" else "false"
    args <- c(args, "--train_model", train_flag)
    
    args
  })
  
  output$cmd_preview <- renderText({
    cmd <- input$python_bin
    args <- build_args()
    paste(cmd, paste(shQuote(args), collapse = " "))
  })
  
  observeEvent(input$run_model, {

    # Validate that requested horizon exists in the converted dataset (if manifest present)
    task_id <- as.character(input$task)
    if (grepl("_h[0-9]+$", task_id)) {
      H <- as.integer(input$pred_horizon)
      hs <- available_horizons()
      if (!is.null(hs) && !(H %in% hs)) {
        log_text(paste0(
          log_text(),
          sprintf("ERROR: Requested horizon H=%d is not present in dataset horizons: %s\n", H, paste(hs, collapse = ","))
        ))
        return(NULL)
      }
      if (is.null(hs)) {
        log_text(paste0(log_text(), "WARNING: horizons manifest not found; training may fail if labels are missing.\n"))
      }
    }

    cmd <- input$python_bin
    args <- build_args()
    full_cmd <- paste(shQuote(cmd), paste(paste0(args), collapse = " "))
    
    log_text("")
    
    tryCatch({
      withProgress(message = "Running model...", value = 0, {
        epochs_total <- NA_real_
        test_batches_total <- NA_real_
        last_progress <- 0
        
        train_fraction <- 0.8
        test_fraction <- 0.2
        
        con <- pipe(full_cmd, open = "r")
        on.exit(close(con), add = TRUE)
        
        repeat {
          line <- readLines(con, n = 1, warn = FALSE)
          if (!length(line)) break
          
          isolate({
            current_log <- log_text()
            log_text(paste0(current_log, line, "\n"))
          })
          
          if (grepl("^DT_PROGRESS_META", line)) {
            m <- regmatches(line, regexec("epochs=([0-9]+)", line))[[1]]
            if (length(m) >= 2) {
              epochs_total <- as.numeric(m[2])
            }
          } else if (grepl("^DT_PROGRESS_EPOCH", line)) {
            m <- regmatches(line, regexec("DT_PROGRESS_EPOCH\\s+([0-9]+)", line))[[1]]
            if (length(m) >= 2 && !is.na(epochs_total) && epochs_total > 0) {
              epoch_num <- as.numeric(m[2])
              new_progress <- min(train_fraction * epoch_num / epochs_total, train_fraction)
              incProgress(
                amount = new_progress - last_progress,
                detail = sprintf("Training epoch %d/%d", epoch_num, epochs_total)
              )
              last_progress <- new_progress
            }
          } else if (grepl("^DT_PROGRESS_TEST_META", line)) {
            m <- regmatches(line, regexec("batches=([0-9]+)", line))[[1]]
            if (length(m) >= 2) {
              test_batches_total <- as.numeric(m[2])
            }
          } else if (grepl("^DT_PROGRESS_TEST_BATCH", line)) {
            m <- regmatches(line, regexec("DT_PROGRESS_TEST_BATCH\\s+([0-9]+)", line))[[1]]
            if (length(m) >= 2 && !is.na(test_batches_total) && test_batches_total > 0) {
              batch_num <- as.numeric(m[2])
              new_progress <- train_fraction +
                min(test_fraction * batch_num / test_batches_total, test_fraction)
              incProgress(
                amount = new_progress - last_progress,
                detail = sprintf("Testing batch %d/%d", batch_num, test_batches_total)
              )
              last_progress <- new_progress
            }
          }
        }
        
        if (last_progress < 1) {
          incProgress(1 - last_progress, detail = "Done.")
        }
      })
    }, error = function(e) {
      isolate({
        current_log <- log_text()
        log_text(paste0(current_log, "Error calling Python: ", conditionMessage(e), "\n"))
      })
    })
    
    run_dir("training_outputs")
  })
  
  output$loss_curves_img <- renderImage({
    file_path <- file.path(run_dir(), "loss_curves.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Loss curves not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "loss_curves.png")
  }, deleteFile = FALSE)
  
  output$cm_img <- renderImage({
    file_path <- file.path(run_dir(), "confusion_matrix.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Confusion matrix not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "confusion_matrix.png")
  }, deleteFile = FALSE)
  
  output$roc_img <- renderImage({
    file_path <- file.path(run_dir(), "roc_curve.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "ROC curve not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "roc_curve.png")
  }, deleteFile = FALSE)
  
  output$cm_test_img <- renderImage({
    file_path <- file.path(run_dir(), "confusion_matrix_test.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Test confusion matrix not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "confusion_matrix_test.png")
  }, deleteFile = FALSE)
  
  output$roc_test_img <- renderImage({
    file_path <- file.path(run_dir(), "roc_curve_test.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Test ROC curve not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "roc_curve_test.png")
  }, deleteFile = FALSE)
  
  output$attn_heatmap_img <- renderImage({
    file_path <- file.path(run_dir(), "attention_heatmap.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Attention heatmap (val) not found yet."))
    }
    list(
      src = file_path,
      contentType = "image/png",
      alt = "attention_heatmap.png",
      style = "max-width: 100%; height: auto;"
    )
  }, deleteFile = FALSE)
  
  output$attn_heatmap_test_img <- renderImage({
    file_path <- file.path(run_dir(), "attention_heatmap_test.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Attention heatmap (test) not found yet."))
    }
    list(
      src = file_path,
      contentType = "image/png",
      alt = "attention_heatmap_test.png",
      style = "max-width: 100%; height: auto;"
    )
  }, deleteFile = FALSE)
  
  observe({
    tasks <- get_available_tasks(input$python_bin)
    
    if (is.null(tasks) || nrow(tasks) == 0) {
      updateSelectInput(
        session, "task",
        choices = c("⚠️ No tasks found" = "")
      )
      return()
    }
    
    choices <- setNames(
      tasks$id,
      paste0(
        tasks$id,
        ifelse(tasks$is_classification, " [CLS]", " [REG]")
      )
    )
    
    updateSelectInput(
      session,
      "task",
      choices = choices,
      selected = tasks$id[1]
    )
  })
}

shinyApp(ui = ui, server = server)