# Correlation Matrix Heatmap Generator
# ----------------------------------
# Generate heatmaps from multiple embryo fate coupling matrices showing correlations between tissues

# Set file paths - modify here to match your matrix file paths
coupling_files <- c(
  "results_embryo1_fate_coupling.csv",
  "results_embryo2_fate_coupling.csv",
  "results_embryo3_fate_coupling.csv"
)

# Load required libraries
library(ggplot2)
library(reshape2)
library(pheatmap)
library(RColorBrewer)
library(gridExtra) 
library(grid)

# Define two tissue clustering approaches
# First approach: group by organ system
tissue_groups_1 <- list(
  "Brain" = c("L brain I", "R brain I", "L brain III", "R brain III"),
  "Gonads" = c("L gonad", "R gonad"),
  "Kidneys" = c("L kidney", "R kidney"),
  "Upper Limbs" = c("L hand", "L arm", "R hand", "R arm"),
  "Lower Limbs" = c("L foot", "L leg", "R foot", "R leg"),
  "Blood" = c("blood")
)

# Second approach: group by germ layer
tissue_groups_2 <- list(
  "Ectoderm" = c("L brain I", "R brain I", "L brain III", "R brain III"),
  "Mesoderm" = c("L gonad", "R gonad", "L kidney", "R kidney", 
                "L hand", "L arm", "R hand", "R arm", 
                "L foot", "L leg", "R foot", "R leg"),
  "Blood" = c("blood")
)

# Define color scheme (same as plot_heatmap.R)
group_colors_1 <- c(
  "Brain" = "#E41A1C",      # Red
  "Gonads" = "#377EB8",     # Blue
  "Kidneys" = "#4DAF4A",    # Green
  "Upper Limbs" = "#984EA3", # Purple
  "Lower Limbs" = "#FF7F00", # Orange
  "Blood" = "#FFFF33"      # Yellow
)

group_colors_2 <- c(
  "Ectoderm" = "#E41A1C",   # Red
  "Mesoderm" = "#377EB8",   # Blue
  "Blood" = "#FFFF33"      # Yellow
)

# Create lists to store results
coupling_matrices <- list()
annotations_1 <- list()
annotations_2 <- list()
heatmaps_1 <- list()
heatmaps_2 <- list()

# Process fate coupling data for each embryo
for (i in 1:length(coupling_files)) {
  coupling_file <- coupling_files[i]
  message(paste("Reading fate coupling data:", coupling_file))
  
  # Read matrix file
  tryCatch({
    data <- read.csv(coupling_file, row.names=1)
  
    # Fix column names - replace dots with spaces to match row names
    colnames(data) <- gsub("\\.", " ", colnames(data))
    
    # Check if data loaded correctly
    if (nrow(data) == 0 || ncol(data) == 0) {
      message(paste("Error: Empty matrix in", coupling_file))
      next
    }

    # Fix diagonal values to exactly 1.0 for correlation matrices
    if (nrow(data) == ncol(data) && all(rownames(data) == colnames(data))) {
      message("Fixing diagonal values to 1.0...")
      diag(data) <- 1.0
    }  

    # Store matrix
    coupling_matrices[[i]] <- data
    
    # Create annotations for first grouping approach
    # Get available tissue names
    available_tissues <- rownames(data)
    
    # First grouping annotation
    tissue_annotation_1 <- data.frame(
      Tissue_Group = rep("Blood", length(available_tissues)),
      row.names = available_tissues
    )
    
    # Assign tissue groups
    for (group_name in names(tissue_groups_1)) {
      matching_tissues <- available_tissues[available_tissues %in% tissue_groups_1[[group_name]]]
      if (length(matching_tissues) > 0) {
        tissue_annotation_1[matching_tissues, "Tissue_Group"] <- group_name
      }
    }
    
    # Second grouping annotation
    tissue_annotation_2 <- data.frame(
      Tissue_Group = rep("Blood", length(available_tissues)),
      row.names = available_tissues
    )
    
    # Assign tissue groups
    for (group_name in names(tissue_groups_2)) {
      matching_tissues <- available_tissues[available_tissues %in% tissue_groups_2[[group_name]]]
      if (length(matching_tissues) > 0) {
        tissue_annotation_2[matching_tissues, "Tissue_Group"] <- group_name
      }
    }
    
    # Store annotations
    annotations_1[[i]] <- tissue_annotation_1
    annotations_2[[i]] <- tissue_annotation_2
    
  }, error = function(e) {
    message(paste("Error processing file", coupling_file, ":", e$message))
  })
}

# Set pheatmap common parameters
# Use same color scheme as plot_heatmap.R for consistency
heatmap_col = rev(c('#08306B','#08519C','#2171B5','#4292C6','#6BAED6','#9ECAE1',
                '#C6DBEF','#DEEBF7','#F7FBFF'))
heatmap_color <- colorRampPalette(heatmap_col)(50)

# Store all heatmap objects
all_heatmaps <- list()

# First clustering approach heatmaps - system grouping
for (i in 1:length(coupling_matrices)) {
  if (!is.null(coupling_matrices[[i]])) {
    # Use first clustering approach
    annotation_colors_1 <- list(
      Tissue_Group = group_colors_1
    )
    
    # Create heatmap title
    title <- paste("Embryo", i, "- Fate Coupling (Tissue Groups)")
    
    # Generate heatmap but don't display
    hm <- pheatmap(
      coupling_matrices[[i]],
      main = title,
      color = heatmap_color,
      border_color = NA,
      cluster_rows = TRUE,
      cluster_cols = TRUE,
      annotation_row = annotations_1[[i]],
      annotation_col = annotations_1[[i]],
      annotation_colors = annotation_colors_1,
      fontsize = 10,
      fontsize_row = 8,
      fontsize_col = 8,
      angle_col = 45,
      legend = TRUE,
      silent = TRUE
    )
    
    # Store heatmap object
    all_heatmaps[[length(all_heatmaps) + 1]] <- hm$gtable
  }
}

# Second clustering approach heatmaps - germ layer grouping
for (i in 1:length(coupling_matrices)) {
  if (!is.null(coupling_matrices[[i]])) {
    # Use second clustering approach
    annotation_colors_2 <- list(
      Tissue_Group = group_colors_2
    )
    
    # Create heatmap title
    title <- paste("Embryo", i, "- Fate Coupling (Germ Layer Groups)")
    
    # Generate heatmap but don't display
    hm <- pheatmap(
      coupling_matrices[[i]],
      main = title,
      color = heatmap_color,
      border_color = NA,
      cluster_rows = TRUE,
      cluster_cols = TRUE,
      annotation_row = annotations_2[[i]],
      annotation_col = annotations_2[[i]],
      annotation_colors = annotation_colors_2,
      fontsize = 10,
      fontsize_row = 8,
      fontsize_col = 8,
      angle_col = 45,
      legend = TRUE,
      silent = TRUE
    )
    
    # Store heatmap object
    all_heatmaps[[length(all_heatmaps) + 1]] <- hm$gtable
  }
}

# Display heatmaps on two pages horizontally
# First page: first three heatmaps (system grouping approach)
if (length(all_heatmaps) >= 3) {
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1, 1)))
  grid.arrange(
    all_heatmaps[[1]], 
    all_heatmaps[[2]], 
    all_heatmaps[[3]], 
    ncol = 3,
    top = textGrob("Fate Coupling - Tissue Groups", gp = gpar(fontsize = 16, font = 2))
  )
}

# Second page: last three heatmaps (germ layer grouping approach)
if (length(all_heatmaps) >= 6) {
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(1, 1)))
  grid.arrange(
    all_heatmaps[[4]], 
    all_heatmaps[[5]], 
    all_heatmaps[[6]], 
    ncol = 3,
    top = textGrob("Fate Coupling - Germ Layer Groups", gp = gpar(fontsize = 16, font = 2))
  )
}



# Set file path for the concatenated MCSA matrix
mcsa_file <- "results_mcsa_matrix.csv"

print("Processing concatenated MCSA matrix from all embryos...")

# Read the MCSA matrix file
tryCatch({
  mcsa_data <- read.csv(mcsa_file, row.names=1)

  # Fix column names - replace dots with spaces to match row names
  colnames(mcsa_data) <- gsub("\\.", " ", colnames(mcsa_data))
  
  # Check if data loaded correctly
  if (nrow(mcsa_data) == 0 || ncol(mcsa_data) == 0) {
    print(paste("Error: Empty matrix in", mcsa_file))
  } else {
    print(paste("Successfully loaded MCSA matrix with dimensions:", nrow(mcsa_data), "x", ncol(mcsa_data)))
    
    # Fix diagonal values to exactly 1.0 for correlation matrices
    if (nrow(mcsa_data) == ncol(mcsa_data) && all(rownames(mcsa_data) == colnames(mcsa_data))) {
      print("Fixing diagonal values to 1.0...")
      diag(mcsa_data) <- 1.0
    }
    
    # Create annotations for MCSA matrix using both grouping approaches
    # Get available tissue names
    available_mcsa_tissues <- rownames(mcsa_data)
    print(paste("Available tissues in MCSA matrix:", paste(available_mcsa_tissues, collapse = ", ")))
    
    # First grouping annotation for MCSA matrix - organ system approach
    mcsa_annotation_1 <- data.frame(
      Tissue_Group = rep("Blood", length(available_mcsa_tissues)),
      row.names = available_mcsa_tissues
    )
    
    # Assign tissue groups for first approach
    for (group_name in names(tissue_groups_1)) {
      matching_tissues <- available_mcsa_tissues[available_mcsa_tissues %in% tissue_groups_1[[group_name]]]
      if (length(matching_tissues) > 0) {
        mcsa_annotation_1[matching_tissues, "Tissue_Group"] <- group_name
        print(paste("Assigned", length(matching_tissues), "tissues to group", group_name, "(System approach)"))
      }
    }
    
    # Second grouping annotation for MCSA matrix - germ layer approach
    mcsa_annotation_2 <- data.frame(
      Tissue_Group = rep("Blood", length(available_mcsa_tissues)),
      row.names = available_mcsa_tissues
    )
    
    # Assign tissue groups for second approach
    for (group_name in names(tissue_groups_2)) {
      matching_tissues <- available_mcsa_tissues[available_mcsa_tissues %in% tissue_groups_2[[group_name]]]
      if (length(matching_tissues) > 0) {
        mcsa_annotation_2[matching_tissues, "Tissue_Group"] <- group_name
        print(paste("Assigned", length(matching_tissues), "tissues to group", group_name, "(Germ layer approach)"))
      }
    }
    
    # Generate MCSA heatmap with first clustering approach (organ system)
    print("Generating MCSA heatmap with organ system grouping...")
    
    annotation_colors_mcsa_1 <- list(
      Tissue_Group = group_colors_1
    )
    
    # Create heatmap for MCSA matrix - system groups
    mcsa_hm_1 <- pheatmap(
      mcsa_data,
      main = "Fate Coupling - All 3 Embryos (Tissue Groups)",
      color = heatmap_color,
      border_color = NA,
      cluster_rows = TRUE,
      cluster_cols = TRUE,
      annotation_row = mcsa_annotation_1,
      annotation_col = mcsa_annotation_1,
      annotation_colors = annotation_colors_mcsa_1,
      fontsize = 12,
      fontsize_row = 10,
      fontsize_col = 10,
      angle_col = 45,
      legend = TRUE,
      silent = TRUE
    )
    
    print("MCSA heatmap with system grouping completed")
    
    # Generate MCSA heatmap with second clustering approach (germ layer)
    print("Generating MCSA heatmap with germ layer grouping...")
    
    annotation_colors_mcsa_2 <- list(
      Tissue_Group = group_colors_2
    )
    
    # Create heatmap for MCSA matrix - germ layer groups
    mcsa_hm_2 <- pheatmap(
      mcsa_data,
      main = "Fate Coupling - All 3 Embryos (Germ Layer Groups)",
      color = heatmap_color,
      border_color = NA,
      cluster_rows = TRUE,
      cluster_cols = TRUE,
      annotation_row = mcsa_annotation_2,
      annotation_col = mcsa_annotation_2,
      annotation_colors = annotation_colors_mcsa_2,
      fontsize = 12,
      fontsize_row = 10,
      fontsize_col = 10,
      angle_col = 45,
      legend = TRUE,
      silent = TRUE
    )
    
    print("MCSA heatmap with germ layer grouping completed")
    
    # Display MCSA heatmaps on a new page
    print("Displaying MCSA heatmaps...")
    
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(1, 1)))
    grid.arrange(
      mcsa_hm_1$gtable, 
      mcsa_hm_2$gtable, 
      ncol = 2,
      top = textGrob("Fate Coupling Matrix - All 100 Embryos", 
                     gp = gpar(fontsize = 18, font = 2))
    )
    
    print("MCSA matrix visualization completed successfully")
  }
  
}, error = function(e) {
  print(paste("Error processing MCSA matrix file", mcsa_file, ":", e$message))
  print("Please ensure the results_mcsa_matrix.csv file exists in the current directory")
})


