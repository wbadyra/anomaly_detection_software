import pandas as pd
import pathlib
from config import *
from ClusteringMethods import ClusteringMethods, plot_2d, plot_3d
from DataPreparation import visualize_data_time
from DimensionReduction import DimensionReduction
from faults_preparation import insert_fault_erratic
from utils import metrics, to_0_1_range, resume_from_previous_fail, found_last_used_param
from clusters_run import run_all_clustering_methods

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.width', 2000)

    params = {}
    
    if RESUME_MODEL_TRAINING:
        last_params = resume_from_previous_fail()
    
    reduction = DimensionReduction()
    
    # methods.visualize_data()
    # visualize_data_time(data=methods.data_drive, params=['Accelerator_Pedal_value', 'Fuel_consumption'])
    # visualize_data_time(data=methods.data_drive, params=['Accelerator Pedal Position [%]', 'MAF [g/s]'])
    # methods.data_correlation()
    for error_cols in COLUMNS_WITH_ERRORS:
        if RESUME_MODEL_TRAINING:
            if not found_last_used_param(current_value=list(error_cols), param_type="columns_with_errors"):
                continue
        params["columns_with_errors"] = list(error_cols)
        for r in REDUCTION_METHODS:
            if RESUME_MODEL_TRAINING:
                if not found_last_used_param(current_value=r, param_type="reduction_method"):
                    continue
            params['reduction_method'] = r
            for errors in ERRORS_TYPES:
                if RESUME_MODEL_TRAINING:
                    if not found_last_used_param(current_value=errors, param_type="errors_types"):
                        continue
                params["errors_types"] = errors
                for dir in DIRECTORIES:
                    if RESUME_MODEL_TRAINING:
                        if not found_last_used_param(current_value=dir, param_type="directories"):
                            continue
                    params["directories"] = dir
                    files = [f for f in pathlib.Path(dir).iterdir() if f.is_file()]
                    for file in list(files):
                        if RESUME_MODEL_TRAINING:
                            print("Retrieved previous parameters!")
                            if not found_last_used_param(current_value=str(file).split('/')[-1], param_type="filename"):
                                continue
                        if 'idle' in str(file).split('/')[-1]:
                            continue
                        print(f"Potencially problematic file: {file}")
                        RESUME_MODEL_TRAINING = False
                        params["filename"] = str(file).split('/')[-1]
                        methods = ClusteringMethods(error_type=errors, filename=file, error_cols=error_cols)
                        # methods.visualize_data_error_comparison(params=['Accelerator_Pedal_value', 'Fuel_consumption'])
                        # methods.visualize_data_error_comparison(params=['Accelerator Pedal Position [%]', 'MAF [g/s]'], period_length=100)
                        # run_all_clustering_methods(methods, error_col=error_cols, reduction=r, visualization=True, params=params)

                        # reduction.dimension_reduction_visualization(data=methods.data.data_drive_scaled_err[methods.data.cols], method='UMAP')
                        # reduction.dimension_reduction_visualization(data=methods.data.data_drive_scaled_err[methods.data.cols], method='PCA')
                        # reduction.dimension_reduction_visualization(data=methods.data.data_drive_scaled_err[methods.data.cols], method='TSNE')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
