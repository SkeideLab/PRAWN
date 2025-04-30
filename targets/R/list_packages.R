# Get a list of all installed packages with their versions
installed_pkgs <- installed.packages()

# Extract only the package names and versions
pkg_versions <- installed_pkgs[, c("Package", "Version")]

# Convert to data frame for easier filtering and readability
pkg_versions_df <- as.data.frame(pkg_versions)

# List of packages you want to check
my_pkg_list <- c("dplyr", 
                 "magrittr",
                 "ggplot2", 
                 "ggpubr",
                 "grid",
                 "png",
                 "readr",
                 "permuco")

# Filter pkg_versions_df based on the list of desired packages
filtered_pkg_versions <- pkg_versions_df[pkg_versions_df$Package %in% my_pkg_list, ]

# Create markdown table
markdown_table <- function(df) {
  # Header row
  cat("| Package | Version |\n")
  cat("|---------|---------|\n")
  
  # Data rows
  apply(df, 1, function(row) {
    cat(sprintf("| %s | %s |\n", row["Package"], row["Version"]))
  })
}

# Call the function to output the markdown table
markdown_table(filtered_pkg_versions)


