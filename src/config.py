import sys, os


# define base directory
if sys.platform.startswith('darwin'):
    base_dir = '/Users/roman/GitHub/PRAWN_ER/'
elif sys.platform.startswith('linux'):
    #base_dir = '/ptmp/kroma/PRAWN/' # R0
    base_dir = '/u/kroma/PRAWN/' # R1
os.chdir(base_dir)


def setup_directories(base_dir):
    """
    Set up directory paths based on the provided base directory.

    Args:
        base_dir (str): Directory of the project.
    
    Returns:
        dict: A dictionary containing various directory paths.
    """
    return {
        'base_dir': base_dir,
        'data_dir': f"{base_dir}data/",
        'raw_dir': f"{base_dir}data/raw/",
        'raw_eeg_dir': f"{base_dir}data/raw/eeg/",
        'raw_video_dir': f"{base_dir}data/raw/videos/ratings/",
        'interim_dir': f"{base_dir}data/interim/",
        'processed_dir': f"{base_dir}data/processed/",
        'plot_dir': f"{base_dir}plots/",
        'ratings_dir': f"{base_dir}data/ratings/",
        'model_dir': f"{base_dir}models/",
        'report_dir': f"{base_dir}reports/",
    }


def create_if_not_exist(dirs):
    """
    Recursively create directories, if not already existent.

    Args:
        dirs (list): List of directories to be created.
    """
    if not isinstance(dirs, list):
        dirs = [dirs]
    for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
            except FileExistsError:
                print(f"Could not create directory {d} because it existed. Maybe it was created by a parallel process.")

def create_sub_ses_dirs(sub_ses_str, dirs):
    """
    Creates the directories for the subject and session in the 
    beginning of the processing pipeline.
    
    Args:
        sub_ses_str (str): The subject and session identifyer.
        dirs (dict): Directory information.
    """
    
    create_if_not_exist([
                     dirs['interim_dir'] + sub_ses_str,
                     dirs['processed_dir'] + sub_ses_str,
                     dirs['model_dir'] + sub_ses_str,
                     dirs['plot_dir'] + sub_ses_str + "/qa",                     
                     dirs['plot_dir'] + sub_ses_str + "/montage",
                     ])
    
    return None


# define subdirectories
dirs = setup_directories(base_dir)

# define some inclusion criteria for a subject
# minimum number of trials per condition
min_trials = 20 

# 1/2 raters agree on attention (liberal), both must agree (conservative)
rating_version = "liberal" 