import os
import csv
from itertools import product
from datetime import datetime
import concurrent.futures
import multiprocessing # <-- import multiprocessing directly

# we need to import the function we want to tune
from process_video import yolo_sahi_pose_tracking

# helper function to run a single sweep combination
def run_single_sweep(params, video_source, video_size, output_video_dir, results_queue):
    # construct a unique output filename for the video based on parameters
    param_string = "_".join(f"{k}_{v}" for k, v in params.items()).replace(".", "p")
    output_video_name = os.path.join(output_video_dir, f"output_video_{param_string}.mp4")
    
    print(f"starting run with parameters: {params} | output: {output_video_name}")

    try:
        # call the video processing function with the current parameters
        # now, yolo_sahi_pose_tracking expects an output_path argument
        yolo_sahi_pose_tracking(
            source=video_source,
            size=video_size,
            output_path=output_video_name, # Pass the unique output path
            **params
        )
        
        row = params.copy()
        row['output_video_path'] = output_video_name
        results_queue.put(row) # Use a queue to safely collect results from processes
        print(f"finished run with parameters: {params}")
        return True # Indicate success
    except Exception as e:
        print(f"error during run with params {params}: {e}")
        # Optionally put an error message in the queue if you want to log failed runs
        error_row = params.copy()
        error_row['output_video_path'] = f"ERROR: {e}"
        results_queue.put(error_row)
        return False # Indicate failure

def run_hyperparameter_sweep():
    # define the video source and size - these will be fixed for all runs
    video_source = 'video raw/test_clip1.mp4'
    video_size = (1440, 810)

    # define the hyperparameters to sweep
    # adjusted parameter ranges for fewer combinations (32 total)
    hyperparameters_to_sweep = {
        'sahi_conf_threshold': [0.1, 0.5], # 2 options
        'sahi_iou_threshold': [0.5, 0.8], # 2 options
        'match_threshold': [0.3, 0.7],    # 2 options
        'iou_weight': [0.35, 0.45, 0.55, 0.65],      # 4 options
    }

    # generate all combinations of hyperparameters
    keys = hyperparameters_to_sweep.keys()
    values = hyperparameters_to_sweep.values()
    hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # prepare to log results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"sweep_results_{timestamp}.csv"
    
    # ensure the output directory for videos exists
    output_video_dir = "sweep_output_videos"
    os.makedirs(output_video_dir, exist_ok=True)

    print(f"starting hyperparameter sweep with {len(hyperparameter_combinations)} combinations...")

    # using multiprocessing.Manager to safely pass results between processes
    manager = multiprocessing.Manager() # <-- Corrected: use multiprocessing.Manager() directly
    results_queue = manager.Queue()

    # use ProcessPoolExecutor to run tasks in parallel
    # max_workers=2 will run 2 combinations at the same time
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_single_sweep, params, video_source, video_size, output_video_dir, results_queue): params 
                   for params in hyperparameter_combinations}

        # write header to CSV immediately
        with open(results_filename, 'w', newline='') as csvfile:
            fieldnames = list(keys) + ['output_video_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            csvfile.flush()

            completed_count = 0
            for future in concurrent.futures.as_completed(futures):
                params_for_future = futures[future]
                try:
                    result = future.result() # This will raise exceptions if the task failed
                    completed_count += 1
                    print(f"[{completed_count}/{len(hyperparameter_combinations)}] combination {params_for_future} processed. collecting results...")
                except Exception as exc:
                    completed_count += 1
                    print(f"[{completed_count}/{len(hyperparameter_combinations)}] combination {params_for_future} generated an exception: {exc}")
            
            # after all futures are completed, collect results from the queue
            print("all tasks completed. writing collected results to csv...")
            # since all workers are done, the queue should not be modified anymore.
            # but it's good practice to ensure workers are properly shut down before draining
            # the queue fully in case there are lingering items.
            while not results_queue.empty():
                row = results_queue.get()
                with open(results_filename, 'a', newline='') as csvfile_append: # open in append mode
                    writer_append = csv.DictWriter(csvfile_append, fieldnames=fieldnames)
                    writer_append.writerow(row)
                    csvfile_append.flush() # ensure data is written immediately
            
    print(f"\nhyperparameter sweep finished. results saved to {results_filename}")
    print(f"check the '{output_video_dir}' directory for output videos.")


if __name__ == '__main__':
    run_hyperparameter_sweep()