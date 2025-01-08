
#Excel file with SB data
datasetSBW <-Sub_data

#package for SOM
install.packages("kohonen")
library(kohonen)

# Scale the numeric data
scaled_data <- scale(datasetSBW)

# Set the size of the grid 
grid_size <- somgrid(xdim = 16, ydim = 16, topo = 'hexagonal', toroidal = T)  # Define the grid explicitly

# Create the SOM
som_model <- som(scaled_data, grid = grid_size, rlen = 40, alpha = c(0.05, 0.01), keep.data = TRUE)


dev.new(width = 10, height = 8)
# Create a blue to red color palette
blue_to_red_palette <- colorRampPalette(c("blue","orange","red"))

# Create SOM visualizations as heatmaps, one per variable
par(mfrow = c(ceiling(ncol(scaled_data) / 5), 5), mar = c(2.5, 2.5, 4.5, 1.5))

for (i in 1:ncol(scaled_data)) {
  variable_heatmap <- matrix(som_model$codes[[1]][, i], nrow = 6, ncol = 6, byrow = TRUE)
  
  # Add heatmap with blue to red color scale
  image(variable_heatmap, col = blue_to_red_palette(50), main = colnames(scaled_data)[i])
}
