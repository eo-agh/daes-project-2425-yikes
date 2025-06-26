import pandas as pd
import os
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.wkt import dumps as wkt_dumps
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- Worker Function ---
# This function processes all 'wezwania' for a single ID_GPS
def process_id_gps(args):
    """
    Processes emergency calls (wezwania) for a single GPS ID, creating a path
    from the start time to the return time.

    Args:
        args (tuple): A tuple containing:
            - id_val (str): The GPS ID to process.
            - temp_df (pd.DataFrame): DataFrame slice containing calls for this ID.
            - all_gps_files (list): List of all GPS file paths.
            - all_gps_files_ids_map (dict): Map of GPS IDs to their file indices.

    Returns:
        pd.DataFrame or None: A DataFrame with processed results for the ID,
                              including 'Line' and 'Line_time' columns,
                              or None if processing fails for this ID.
    """
    id_val, temp_df, all_gps_files, all_gps_files_ids_map = args
    results_list = [] # Store results as list of dicts for efficiency

    indexes = all_gps_files_ids_map.get(id_val, [])
    if not indexes:
        return None # Indicate no data found or issue

    # Read and concatenate all GPS files for the current ID
    try:
        required_cols = ['GPS_TIME', 'GPS_LON', 'GPS_LAT']
        gps_df_list = []
        for index in indexes:
            try:
                df_part = pd.read_parquet(all_gps_files[index], columns=required_cols)
                gps_df_list.append(df_part)
            except Exception as read_err:
                print(f"Warning: Error reading file {all_gps_files[index]} for ID {id_val}: {read_err}")

        if not gps_df_list:
                 print(f"Warning: No GPS files successfully read for ID {id_val}.")
                 return None

        gps_df = pd.concat(gps_df_list, ignore_index=True)

    except Exception as e:
        print(f"Error reading or concatenating GPS files for ID {id_val}: {e}")
        return None

    if gps_df.empty:
        return None

    if not all(col in gps_df.columns for col in required_cols):
        print(f"Error: Missing required columns in loaded GPS data for ID {id_val}. Need {required_cols}")
        return None

    # --- Data Cleaning and Preparation ---
    try:
        if not pd.api.types.is_datetime64_any_dtype(gps_df['GPS_TIME']):
            gps_df['GPS_TIME'] = pd.to_datetime(gps_df['GPS_TIME'], errors='coerce')
            gps_df.dropna(subset=['GPS_TIME'], inplace=True)

        if gps_df.empty:
                print(f"Warning: No valid GPS_TIME entries after conversion for ID {id_val}.")
                return None

        gps_df['GPS_LON'] = pd.to_numeric(gps_df['GPS_LON'], errors='coerce')
        gps_df['GPS_LAT'] = pd.to_numeric(gps_df['GPS_LAT'], errors='coerce')
        gps_df.dropna(subset=['GPS_LON', 'GPS_LAT'], inplace=True)

        if gps_df.empty:
                print(f"Warning: No valid GPS coordinates after cleaning for ID {id_val}.")
                return None

        gps_df.sort_values(by=['GPS_TIME'], inplace=True, ignore_index=True)

        gps_time_array = gps_df['GPS_TIME'].to_numpy() # Numpy datetime64 array
        gps_lat_array = gps_df['GPS_LAT'].to_numpy()
        gps_lon_array = gps_df['GPS_LON'].to_numpy()

        # Keep GeoSeries for creating the final LineString
        gps_points_geoseries = gpd.GeoSeries(
            [Point(lon, lat) for lon, lat in zip(gps_lon_array, gps_lat_array)],
            index=gps_df.index # Align index with gps_df
        )
        gps_points_geoseries.set_crs("EPSG:4326", inplace=True)

    except Exception as prep_err:
        print(f"Error preparing GPS data for ID {id_val}: {prep_err}")
        return None


    # --- Process each 'wezwanie' for this ID ---
    for _, wezwanie_row in temp_df.iterrows():
        line, line_time = None, [] # Default values
        start_time_idx = -1        # Index of the GPS point at or just before Czas wezwania
        end_time_idx = -1          # Index of the GPS point just before Czas powrotu ZRM

        try:
            czas_wezwania = pd.to_datetime(wezwanie_row['Czas wezwania'])
            if pd.isna(czas_wezwania):
                # print(f"Skipping row due to missing 'Czas wezwania' for ID {id_val}")
                continue

            # Get target coordinates
            try:
                longitude = float(wezwanie_row['Dlugość geograficzna'])
                latitude = float(wezwanie_row['Szerokość geograficzna'])
                # cel_coords = (latitude, longitude) # Kept for context, not used for LineString
            except (ValueError, TypeError) as coord_err:
                pass
            czas_powrotu = pd.to_datetime(wezwanie_row['Czas powrotu ZRM'])
            if pd.isna(czas_powrotu):
                # print(f"Skipping row due to missing 'Czas powrotu ZRM' for ID {id_val}")
                continue

            # --- Core Logic: Find Path ---
            # Find the index of the GPS point at or just before Czas wezwania
            earlier_times_mask = gps_time_array <= np.datetime64(czas_wezwania)
            if earlier_times_mask.any():
                start_time_idx = np.argwhere(earlier_times_mask).flatten()[-1]
            else:
                # print(f"No GPS point found at or before Czas wezwania for ID {id_val}, Czas wezwania {czas_wezwania}")
                continue # Skip this wezwanie if no valid start point found

            # Find the index of the *last* GPS point strictly *before* Czas powrotu ZRM
            later_times_mask = gps_time_array < np.datetime64(czas_powrotu)
            # We only care about points *after or at* the start index
            valid_end_point_candidates = np.where(later_times_mask & (np.arange(len(gps_time_array)) >= start_time_idx))[0]

            if len(valid_end_point_candidates) > 0:
                end_time_idx = valid_end_point_candidates[-1]
            else:
                pass


            # --- Create LineString using the found indices ---
            # Ensure we have valid start and end indices, and start is not after end
            if start_time_idx != -1 and end_time_idx != -1 and start_time_idx <= end_time_idx:
                # Slice includes start index, excludes end index, so add 1 to end_time_idx
                line_indices = range(start_time_idx, end_time_idx + 1)

                # Check if line has at least 2 points (required for LineString)
                if len(line_indices) >= 2:
                    # Use the original GeoSeries with Shapely points to build the LineString
                    line_geom_points = gps_points_geoseries.iloc[line_indices].tolist()
                    line = LineString(line_geom_points)
                    line_time = gps_time_array[line_indices].tolist() # Get corresponding times

        except Exception as row_err:
            # Add more context to the error message
            wezwanie_info = wezwanie_row.get('Czas wezwania', 'N/A')
            print(f"Error processing row for ID {id_val}, Czas wezwania {wezwanie_info}: {row_err}")
            # Continue to next row, but keep defaults (line=None, line_time=[])

        # Append result for this wezwanie_row
        output_row_dict = wezwanie_row.to_dict()
        # Store Shapely object first
        output_row_dict['Line'] = line
        output_row_dict['Line_time'] = line_time # Store list of times
        results_list.append(output_row_dict)

    if not results_list:
        return None # No successful rows processed for this ID

    return pd.DataFrame(results_list) # Convert list of dicts to DataFrame at the end

# --- Main Execution Block ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # Needed for Windows executable support

    print("Starting script...")
    # --- Your existing setup code ---
    try:
        # Make sure the correct input file is specified
        df = pd.read_parquet('kzw.parquet') # Or 'kzw_powroty_zrm.parquet' if that's the input
        print(f"Loaded main dataframe: {len(df)} rows.")
    except FileNotFoundError:
        print("Error: Input parquet file (e.g., kzw.parquet) not found.")
        exit()
    except Exception as e:
        print(f"Error loading input parquet file: {e}")
        exit()

    # Ensure required columns exist in the input dataframe
    required_input_cols = ['ID_GPS', 'Czas wezwania', 'Dlugość geograficzna', 'Szerokość geograficzna', 'Czas powrotu ZRM']
    if not all(col in df.columns for col in required_input_cols):
        missing_cols = [col for col in required_input_cols if col not in df.columns]
        print(f"Error: Input dataframe is missing required columns: {missing_cols}")
        exit()


    df['ID_GPS'] = df['ID_GPS'].astype('string')
    unique_id_gps = df['ID_GPS'].unique()
    print(f"Found {len(unique_id_gps)} unique GPS IDs.")

    # Adjust folder paths as needed
    # folders = ['gps/2021_I_kw', 'gps/2021_II_kw', 'gps/2021_III_kw', 'gps/2021_IV_kw']
    folders = ['2021_IV_kw'] # Make sure this path exists and contains your GPS files =============================================================================================================
    all_gps_files = []
    print("Scanning for GPS files...")
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"Warning: Folder not found - {folder}")
            continue
        try:
            files = os.listdir(folder)
            for file in files:
                if file.endswith('.parquet'):
                    all_gps_files.append(os.path.join(folder, file))
        except Exception as e:
                 print(f"Error listing files in folder {folder}: {e}")

    if not all_gps_files:
        print("Error: No GPS parquet files found in the specified folders.")
        exit()
    print(f"Found {len(all_gps_files)} GPS parquet files.")

    # Build map from file ID to list of file paths indices
    all_gps_files_ids_map = {value: [] for value in unique_id_gps}
    all_gps_files_paths_map = {value: [] for value in unique_id_gps} # Store paths for easier debugging
    count_matched_files = 0
    for i, file_path in enumerate(all_gps_files):
        try:
            filename = os.path.basename(file_path)
            file_id = filename.split('__')[0]

            if file_id in all_gps_files_ids_map:
                all_gps_files_ids_map[file_id].append(i)
                all_gps_files_paths_map[file_id].append(file_path) # Store path
                count_matched_files += 1
        except IndexError:
            print(f"Warning: Could not extract ID from filename format: {filename}")
        except Exception as e:
            print(f"Error processing filename {file_path}: {e}")

    print(f"Matched {count_matched_files} GPS files to IDs found in the main dataframe.")

    # --- Prepare arguments for parallel processing ---
    tasks = []
    print("Preparing tasks...")
    for id_val in unique_id_gps:
        # Check if ID has corresponding files *and* exists in the main dataframe
        if id_val in all_gps_files_ids_map and all_gps_files_ids_map[id_val]:
            temp_df = df[df['ID_GPS'] == id_val].copy()
            if not temp_df.empty:
                tasks.append((id_val, temp_df, all_gps_files, all_gps_files_ids_map))
        # else:
            # Optional: Log skipped IDs
            # print(f"Skipping ID {id_val} due to no matching GPS files or no entries in main df.")


    print(f"Prepared {len(tasks)} tasks for parallel processing.")
    if not tasks:
        print("No tasks to process. Exiting.")
        exit()

    # --- Parallel Execution ---
    results = []
    # Use cpu_count() or specify a number, leave one core free if needed
    num_processes = max(1, multiprocessing.cpu_count() - 4)
    # Ensure at least one process
    if num_processes < 1:
           num_processes = 1
    print(f"Starting parallel processing with {num_processes} workers...")

    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use tqdm for progress bar with imap_unordered
            results = list(tqdm(pool.imap_unordered(process_id_gps, tasks), total=len(tasks)))
    except Exception as pool_err:
        print(f"\nError during parallel processing pool execution: {pool_err}")
        exit() # Exit if the pool itself fails critically

    print("Parallel processing finished.")

    # --- Combine Results ---
    final_results_list = [res_df for res_df in results if res_df is not None and not res_df.empty]

    if not final_results_list:
        print("No results were successfully processed or returned from workers.")
        exit()

    print(f"Combining results from {len(final_results_list)} successful tasks...")
    try:
        dojazdy_df = pd.concat(final_results_list, ignore_index=True)
    except Exception as concat_err:
        print(f"Error during final concatenation of results: {concat_err}")
        exit()

    print(f"Combined DataFrame shape: {dojazdy_df.shape}")

    # --- Final Conversion and Saving ---
    print("Converting LineString geometry to WKT for saving...")
    try:
        # Ensure 'Line' column exists before applying conversion
        if 'Line' in dojazdy_df.columns:
            # Use optimized WKT conversion
            dojazdy_df['Line'] = dojazdy_df['Line'].apply(lambda x: wkt_dumps(x) if x and isinstance(x, LineString) else None)
        else:
            print("Warning: 'Line' column not found in the final DataFrame. Skipping WKT conversion.")

        # Convert Line_time (list of datetime objects or numpy.datetime64) to list of strings
        if 'Line_time' in dojazdy_df.columns:
             print("Converting Line_time to string format...")
             dojazdy_df['Line_time'] = dojazdy_df['Line_time'].apply(
                 lambda time_list: [str(pd.to_datetime(t)) for t in time_list] # Ensure conversion to standard datetime string
                 if time_list and isinstance(time_list, list) and len(time_list) > 0 else None
             )
        else:
            print("Warning: 'Line_time' column not found in the final DataFrame. Skipping time conversion.")

    except Exception as convert_err:
        print(f"Error during final conversion (WKT or Line_time): {convert_err}")
        exit()


    print(f"Finished processing. Saving {len(dojazdy_df)} results.")
    output_filename = 'dojazdy_df_parallel_full_path.parquet'
    try:
        dojazdy_df.to_parquet(output_filename, index=False)
        print(f"Successfully saved results to {output_filename}")
    except Exception as e:
        print(f"Error saving results to parquet: {e}")

    print("Script finished.")