# Here I need to somehow group variables used

kia_soul_dir = "/home/wasyl/magisterka/fault_detection/processed_data/datasets/Kia_soul/"
OBD_II_dir = "/home/wasyl/magisterka/fault_detection/processed_data/datasets/OBD-II-Dataset/"
archive_dir = "/home/wasyl/magisterka/fault_detection/processed_data/datasets/archive/"
dataset_dir = "/home/wasyl/magisterka/fault_detection/processed_data/datasets/Dataset/"
carOBD_dir = "/home/wasyl/magisterka/fault_detection/processed_data/datasets/carOBD-master/"

COLUMNS_WITH_ERRORS = [['Engine RPM [RPM]'],['Engine RPM [RPM]', 'Intake Manifold Pressure [kPa]', 'Engine Coolant Temperature [C]']]
ERRORS_TYPES = ["erratic", "spike", "drift", "hardover"]
DIRECTORIES = [OBD_II_dir, dataset_dir, kia_soul_dir, carOBD_dir]
REDUCTION_METHODS =  ['PCA']

RESUME_MODEL_TRAINING = False

dir_names_dict = {kia_soul_dir: 'kia_soul',
                  OBD_II_dir: 'OBD_II',
                  dataset_dir: 'dataset',
                  carOBD_dir: 'carOBD',
                  archive_dir: 'archive'}