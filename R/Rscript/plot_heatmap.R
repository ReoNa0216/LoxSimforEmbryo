# MCSA Tissue-Barcode Heatmap Generator 
library(ggplot2)
library(pheatmap)
library(gridExtra)
library(RColorBrewer)
library(grid)

# --- 1. Configuration ---

# File path 
concatenated_file <- "results_concatenated_tissue_barcode_matrix.csv"

# Output PDF path
output_pdf <- "~/Rplot/HeatmapE12.5_standard/heatmap_results.pdf"

# Tissue grouping definitions
tissue_groups_1 <- list(
  "Brain" = c("L brain I", "R brain I", "L brain III", "R brain III"),
  "Gonads" = c("L gonad", "R gonad"),
  "Kidneys" = c("L kidney", "R kidney"),
  "Upper Limbs" = c("L hand", "L arm", "R hand", "R arm"),
  "Lower Limbs" = c("L foot", "L leg", "R foot", "R leg"),
  "Blood" = c("blood")
)
tissue_groups_2 <- list(
  "Ectoderm" = c("L brain I", "R brain I", "L brain III", "R brain III"),
  "Mesoderm" = c("L gonad", "R gonad", "L kidney", "R kidney", 
                 "L hand", "L arm", "R hand", "R arm", 
                 "L foot", "L leg", "R foot", "R leg"),
  "Blood" = c("blood")
)

# Color schemes
group_colors_1 <- setNames(brewer.pal(length(tissue_groups_1), "Set2"), names(tissue_groups_1))
group_colors_2 <- setNames(brewer.pal(length(tissue_groups_2), "Dark2"), names(tissue_groups_2))
heatmap_color <- colorRampPalette(rev(c('#08306B','#08519C','#2171B5','#4292C6','#6BAED6','#9ECAE1',
                                        '#C6DBEF','#DEEBF7','#F7FBFF')))(100)

# --- 2. Helper Functions ---

process_matrix <- function(file_path) {
  if (!file.exists(file_path)) return(NULL)
  data <- read.csv(file_path, row.names = 1, check.names = FALSE)
  data <- data[rowSums(data) > 0, ]
  data <- data[, colSums(data) > 0]
  data_norm <- apply(data, 2, function(x) if(sum(x) > 0) x / sum(x) else x)
  return(data_norm)
}

create_row_annotation <- function(tissues, groups, group_colors) {
  annotation_df <- data.frame(Group = character(length(tissues)), row.names = tissues)
  for (group_name in names(groups)) {
    matching_tissues <- tissues[tissues %in% groups[[group_name]]]
    if (length(matching_tissues) > 0) {
      annotation_df[matching_tissues, "Group"] <- group_name
    }
  }
  annotation_df$Group[annotation_df$Group == ""] <- names(groups)[1]
  
  return(list(
    df = annotation_df,
    colors = list(Group = group_colors)
  ))
}

create_aggregated_matrix <- function(data_norm, groups) {
  aggregated_data <- matrix(0, nrow = length(groups), ncol = ncol(data_norm))
  rownames(aggregated_data) <- names(groups)
  colnames(aggregated_data) <- colnames(data_norm)
  
  for (group_name in names(groups)) {
    tissues_in_group <- groups[[group_name]]
    tissues_in_data <- intersect(tissues_in_group, rownames(data_norm))
    if (length(tissues_in_data) > 0) {
      aggregated_data[group_name, ] <- if (length(tissues_in_data) == 1) data_norm[tissues_in_data, ] else colSums(data_norm[tissues_in_data, ])
    }
  }
  return(aggregated_data)
}

# --- 3. Main Processing ---

# Process concatenated file
data_concat <- process_matrix(concatenated_file)
aggregated_data <- create_aggregated_matrix(data_concat, tissue_groups_2)

# Perform K-means clustering (adjust centers as needed)
set.seed(123)
centers_num <- 8  # Adjust this number as needed
kmeans_result_agg <- kmeans(t(aggregated_data), centers = centers_num, nstart = 25, 
                            iter.max = 300)

# --- MANUAL CLUSTER ASSIGNMENT ---
manual_clusters <- kmeans_result_agg$cluster
manual_clusters[manual_clusters %in% c(4,1,7)] <- "Ectoderm"
manual_clusters[manual_clusters %in% c(3,5)] <- "Mesoderm"
manual_clusters[manual_clusters %in% c(6)] <- "Blood"
manual_clusters[manual_clusters %in% c(2,8)] <- "Bilineage"

# manual_clusters <- as.factor(kmeans_result_agg$cluster)
manual_clusters <- as.factor(manual_clusters)

# Create column annotation
col_annotation <- data.frame(Cluster = manual_clusters)
rownames(col_annotation) <- colnames(aggregated_data)
col_order <- order(manual_clusters)

# Define cluster colors
cluster_colors <- setNames(brewer.pal(length(unique(manual_clusters)), "Accent"), 
                           unique(manual_clusters))
col_annotation_colors <- list(Cluster = cluster_colors)

# Determine column gaps
col_gaps <- cumsum(table(manual_clusters)[unique(manual_clusters[col_order])])[-length(unique(manual_clusters))]

# --- 4. Plotting Function ---

create_heatmap <- function(data_matrix, title, groups, group_colors, is_aggregated = FALSE, custom_col_order = NULL) {
  # Use custom column order if provided, otherwise use global col_order
  current_col_order <- if (!is.null(custom_col_order)) custom_col_order else col_order
  current_col_gaps <- if (!is.null(custom_col_order)) {
    # Recalculate gaps based on current data
    current_clusters <- col_annotation[colnames(data_matrix), "Cluster"]
    cumsum(table(current_clusters)[unique(current_clusters[current_col_order])])[-length(unique(current_clusters))]
  } else col_gaps
  
  if (is_aggregated) {
    # For aggregated data
    agg_row_names <- rownames(data_matrix)
    agg_annotation_df <- data.frame(Group = agg_row_names, row.names = agg_row_names)
    agg_annotation_colors <- list(Group = group_colors)
    
    # Current column annotation for this subset
    current_col_annotation <- col_annotation[colnames(data_matrix), , drop = FALSE]
    
    pheatmap(data_matrix[, current_col_order], main = title,
             color = heatmap_color, border_color = NA,
             cluster_rows = FALSE, cluster_cols = FALSE,
             annotation_row = agg_annotation_df, 
             annotation_col = current_col_annotation,
             annotation_colors = append(agg_annotation_colors, col_annotation_colors),
             gaps_col = current_col_gaps,
             show_colnames = FALSE, legend = TRUE, silent = TRUE)
  } else {
    # For standard data
    annotation <- create_row_annotation(rownames(data_matrix), groups, group_colors)
    current_col_annotation <- col_annotation[colnames(data_matrix), , drop = FALSE]
    
    pheatmap(data_matrix[, current_col_order], main = title,
             color = heatmap_color, border_color = NA,
             cluster_rows = FALSE, cluster_cols = FALSE, 
             annotation_row = annotation$df, 
             annotation_col = current_col_annotation, 
             annotation_colors = append(annotation$colors, col_annotation_colors),
             gaps_col = current_col_gaps,
             show_colnames = FALSE, legend = TRUE, silent = TRUE)
  }
}

# --- 5. Generate and Save Plots ---

# Start PDF device
pdf(output_pdf, width = 12, height = 8)

# 1. Concatenated - Aggregated by Germ Layer
p_concat_agg <- create_heatmap(aggregated_data, "Concatenated - Aggregated by Germ Layer", 
                               tissue_groups_2, group_colors_2, is_aggregated = TRUE)
grid.draw(p_concat_agg$gtable)

# 2. Extract embryo-specific barcodes and create aggregated plots
for (embryo_num in 1:3) {
  embryo_pattern <- paste0("^embryo", embryo_num)
  embryo_cols <- grep(embryo_pattern, colnames(aggregated_data), value = TRUE)
  
  if (length(embryo_cols) > 0) {
    # Map to original cluster order
    embryo_indices <- which(colnames(aggregated_data) %in% embryo_cols)
    embryo_order <- intersect(col_order, embryo_indices)
    
    if (length(embryo_order) > 0) {
      embryo_data <- aggregated_data[, embryo_cols, drop = FALSE]
      
      # Create embryo-specific column order
      embryo_clusters <- col_annotation[embryo_cols, "Cluster"]
      embryo_col_order <- order(embryo_clusters)
      
      p_embryo_agg <- create_heatmap(embryo_data, paste("Embryo", embryo_num, "- Aggregated by Germ Layer"),
                                     tissue_groups_2, group_colors_2, is_aggregated = TRUE, 
                                     custom_col_order = embryo_col_order)
      grid.newpage()
      grid.draw(p_embryo_agg$gtable)
    }
  }
}

# 3. Concatenated - Standard heatmaps
# Tissue Groups
p_concat_tissue <- create_heatmap(data_concat, "Concatenated - Tissue Groups", 
                                  tissue_groups_1, group_colors_1, is_aggregated = FALSE)
grid.newpage()
grid.draw(p_concat_tissue$gtable)

# Germ Layer Groups  
p_concat_germ <- create_heatmap(data_concat, "Concatenated - Germ Layer Groups", 
                                tissue_groups_2, group_colors_2, is_aggregated = FALSE)
grid.newpage()
grid.draw(p_concat_germ$gtable)

# 4. Extract embryo-specific barcodes for standard plots
for (embryo_num in 1:3) {
  embryo_pattern <- paste0("^embryo", embryo_num)
  embryo_cols <- grep(embryo_pattern, colnames(data_concat), value = TRUE)
  
  if (length(embryo_cols) > 0) {
    embryo_data <- data_concat[, embryo_cols, drop = FALSE]
    
    # Create embryo-specific column order
    embryo_clusters <- col_annotation[embryo_cols, "Cluster"]
    embryo_col_order <- order(embryo_clusters)
    
    # Tissue Groups
    p_embryo_tissue <- create_heatmap(embryo_data, paste("Embryo", embryo_num, "- Tissue Groups"),
                                      tissue_groups_1, group_colors_1, is_aggregated = FALSE,
                                      custom_col_order = embryo_col_order)
    grid.newpage()
    grid.draw(p_embryo_tissue$gtable)
    
    # Germ Layer Groups
    p_embryo_germ <- create_heatmap(embryo_data, paste("Embryo", embryo_num, "- Germ Layer Groups"),
                                    tissue_groups_2, group_colors_2, is_aggregated = FALSE,
                                    custom_col_order = embryo_col_order)
    grid.newpage()
    grid.draw(p_embryo_germ$gtable)
  }
}

# Close PDF device
dev.off()

