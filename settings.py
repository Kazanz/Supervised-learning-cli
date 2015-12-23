import os

DELIMITER = " "
TRAINING_CSV = os.environ.get("TRAINING_CSV", 'LTV_training.csv')
PLOT_FILE = os.environ.get("PLOT_FILE", 'LTV_plots.pdf')
