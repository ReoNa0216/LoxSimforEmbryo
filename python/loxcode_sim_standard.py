import numpy as np
import random
import collections
import os
import datetime  
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import time as py_time  
from typing import List, Dict, Tuple, Set, Optional
from tqdm import tqdm
# Add Numba for JIT compilation  
from numba import jit

# Helper function for directory creation
def create_date_directory():
    """
    Create a directory with today's date and timestamp to ensure uniqueness
    
    Returns:
        str: Path to the newly created directory
    """
    # Format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = timestamp
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# JIT-accelerated helper function for fate determination
@jit(nopython=True)
def calculate_fate_decisions(current_states, split_times, time, diff_speed, transition_matrix):
    """
    Accelerated batch processing of cell fate decisions
    
    Parameters:
        current_states: Array of current cell states
        split_times: Array of split times for each state
        time: Current simulation time
        diff_speed: Differentiation speed parameter
        transition_matrix: Probability matrix for state transitions
    
    Returns:
        Array of new states (same as input if no change)
    """
    new_states = np.zeros_like(current_states)
    for i, state in enumerate(current_states):
        if state >= 16:  # Already terminal cell type
            new_states[i] = state
            continue
            
        if split_times[state] < time and np.random.random() < diff_speed:
            # Select transition based on probability
            probs = transition_matrix[state]
            # Normalize to ensure sum is 1.0
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs = probs / prob_sum
            
            # Draw random sample according to probabilities
            r = np.random.random()
            cumsum = 0.0
            for j, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    new_states[i] = j
                    break
            if r >= cumsum:  # Safety check
                new_states[i] = state  # No change
        else:
            new_states[i] = state  # No change
            
    return new_states

# Helper function added outside the class
def _simulate_single_embryo(params):
    """Helper function for simulating a single embryo that can be pickled by multiprocessing"""
    embryo_idx, num_embryos, t_barcoding, t_collection, times, fluxes, diff_speed, lox_original, output_dir = params
    
    print(f"\nStarting simulation for embryo #{embryo_idx+1}/{num_embryos}")
    
    # Only create detailed output files for first 3 embryos
    if embryo_idx < 3:
        # Create unique output filenames for each embryo
        output_suffix = f"_embryo{embryo_idx+1}"
        pedigree_file = os.path.join(output_dir, f"loxcode_full_pedigree{output_suffix}.csv")
        barcoding_file = os.path.join(output_dir, f"loxcode_census_at_barcoding{output_suffix}.csv")
        sampling_file = os.path.join(output_dir, f"loxcode_census_at_sampling{output_suffix}.csv")
    else:
        # For embryos 4 and above, only create sampling file with temporary name
        pedigree_file = None
        barcoding_file = None
        sampling_file = os.path.join(output_dir, f"loxcode_census_at_sampling_embryo{embryo_idx+1}.csv")
    
    # Create new model instance
    model = LoxCodeEmbryoModel()
    model.times = times.copy()
    model.fluxes = fluxes.copy()
    model.diff_speed = diff_speed
    model.lox_original = lox_original.copy()
    model.split_times = model._calculate_split_times()
    model.transition_matrix = model._create_transition_matrix()
    
    # Simulate single embryo
    model.sim_single(t_barcoding, t_collection, 
                  pedigree_file=pedigree_file,
                  barcoding_file=barcoding_file,
                  sampling_file=sampling_file)
    
    return sampling_file
    
class LoxCodeEmbryoModel:
    """
    Agent-based embryo development model for simulating LoxCode barcoding and cell fate determination processes
    """
    def __init__(self):
        # Default parameters
        self.times = np.array([139.9, 28.6, 1.1, 5.8, 5.3, 21.4, 
                              0.5, 2.9, 2.4, 1.4, 7.5, 20.5, 
                              25.1, 2.1, 34.5, 7.5])
        self.fluxes = np.array([17.3, 75.1, 29.6, 53.8, 59.9, 30.7, 
                               37.7, 51.3, 48.5, 59.0, 41.3, 47.5, 
                               77.6, 44.4, 47.9, 38.5]) / 100.0  # Convert to 0-1 probability
        self.diff_speed = 1.0 / 5.5
        self.lox_original = list(range(1, 14))  # Original barcode sequence [1,2,3,...,13]
        
        # Calculate division times
        self.split_times = self._calculate_split_times()
        
        # Create transition matrix
        self.transition_matrix = self._create_transition_matrix()
        
    def _calculate_split_times(self):
        """Calculate division time for each cell type"""
        split_times = np.zeros(16)
        split_times[0] = self.times[0]
        split_times[1] = split_times[0] + self.times[1]
        split_times[2] = split_times[1] + self.times[2]
        split_times[3] = split_times[2] + self.times[3]
        split_times[4] = split_times[2] + self.times[4]
        split_times[5] = split_times[1] + self.times[5]
        split_times[6] = split_times[5] + self.times[6]
        split_times[7] = split_times[6] + self.times[7]
        split_times[8] = split_times[6] + self.times[8]
        split_times[9] = split_times[5] + self.times[9]
        split_times[10] = split_times[9] + self.times[10]
        split_times[11] = split_times[10] + self.times[11]
        split_times[12] = split_times[10] + self.times[12]
        split_times[13] = split_times[9] + self.times[13]
        split_times[14] = split_times[13] + self.times[14]
        split_times[15] = split_times[13] + self.times[15]
        return split_times
    
    def _create_transition_matrix(self):
        """Create cell fate transition matrix"""
        dim = 33
        D = np.zeros((dim, dim))
        
        # Set transition probabilities based on fluxes parameters
        D[0, 1] = self.fluxes[0]
        D[0, 16] = 1 - D[0, 1]
        D[1, 2] = self.fluxes[1]
        D[1, 5] = 1 - D[1, 2]
        D[2, 3] = self.fluxes[2]
        D[2, 4] = 1 - D[2, 3]
        D[3, 17] = self.fluxes[3]
        D[3, 18] = 1 - D[3, 17]
        D[4, 19] = self.fluxes[4]
        D[4, 20] = 1 - D[4, 19]
        D[5, 6] = self.fluxes[5]
        D[5, 9] = 1 - D[5, 6]
        D[6, 7] = self.fluxes[6]
        D[6, 8] = 1 - D[6, 7]
        D[7, 21] = self.fluxes[7]
        D[7, 22] = 1 - D[7, 21]
        D[8, 23] = self.fluxes[8]
        D[8, 24] = 1 - D[8, 23]
        D[9, 10] = self.fluxes[9]
        D[9, 13] = 1 - D[9, 10]
        D[10, 11] = self.fluxes[10]
        D[10, 12] = 1 - D[10, 11]
        D[11, 25] = self.fluxes[11]
        D[11, 26] = 1 - D[11, 25]
        D[12, 27] = self.fluxes[12]
        D[12, 28] = 1 - D[12, 27]
        D[13, 14] = self.fluxes[13]
        D[13, 15] = 1 - D[13, 14]
        D[14, 29] = self.fluxes[14]
        D[14, 30] = 1 - D[14, 29]
        D[15, 31] = self.fluxes[15]
        D[15, 32] = 1 - D[15, 31]
        
        return D

    def is_odd(self, a):
        """Check if a number is odd"""
        return a % 2 != 0
    
    def generate_codes(self, N, P, lox_size):
        """
        Generate LoxCode barcodes
        
        Parameters:
        N: Number of barcodes to generate
        P: Average number of recombination events
        lox_size: Number of loxP sites
        
        Returns:
        List of generated barcodes
        """
        min_length = 82
        lox = 34
        elem = 7
        
        np.random.seed(int(py_time.time()))
        codes = {}
        multi_codes = []
        
        for _ in range(N[0]):
            cassette = list(range(1, lox_size + 1))
            n_rec = P[0]
            actual_rec = 0
            stored = False
            
            for _ in range(int(n_rec) + 1):
                if len(cassette) == 1:
                    code_tuple = tuple(cassette)
                    if code_tuple in codes:
                        codes[code_tuple] += 1
                    else:
                        codes[code_tuple] = 1
                    multi_codes.append(cassette)
                    stored = True
                    break
                
                if (len(cassette) - 2) * lox + (len(cassette) - 1) * elem < min_length:
                    code_tuple = tuple(cassette)
                    if code_tuple in codes:
                        codes[code_tuple] += 1
                    else:
                        codes[code_tuple] = 1
                    multi_codes.append(cassette)
                    stored = True
                    break
                
                pos1 = np.random.randint(0, len(cassette))
                pos2 = np.random.randint(0, len(cassette))
                
                if abs(pos1 - pos2) * lox + (abs(pos1 - pos2) + 1) * elem < min_length:
                    continue
                
                if self.is_odd(pos1) == self.is_odd(pos2):
                    dice = np.random.random()
                    if dice < 1:
                        # Invert elements in the interval and make them negative
                        for j in range(min(pos1, pos2), max(pos1, pos2) + 1):
                            cassette[j] *= -1
                        # Reverse the sequence
                        cassette[min(pos1, pos2):max(pos1, pos2) + 1] = cassette[min(pos1, pos2):max(pos1, pos2) + 1][::-1]
                        actual_rec += 1
                    else:
                        continue
                else:
                    # Excise elements in the interval
                    cassette = cassette[:min(pos1, pos2)] + cassette[max(pos1, pos2) + 1:]
                    actual_rec += 1
            
            if not stored:
                code_tuple = tuple(cassette)
                if code_tuple in codes:
                    codes[code_tuple] += 1
                else:
                    codes[code_tuple] = 1
                multi_codes.append(cassette)
                
        return multi_codes

    def sim_single(self, t_barcoding=132, t_collection=228,
                pedigree_file=None, barcoding_file=None, sampling_file=None, output_dir=None):
        """
        Simulate the development of a single embryo from 4-cell stage to specified time
        
        Parameters:
        t_barcoding: Barcoding time (hours)
        t_collection: Sampling time (hours)
        pedigree_file: Pedigree file name (None means don't save)
        barcoding_file: Census at barcoding time file name (None means don't save)
        sampling_file: Census at sampling time file name
        output_dir: Directory to save output files
        """
        # This prevents creating empty directories in multiprocessing mode
        if output_dir is None and sampling_file is None:
            output_dir = create_date_directory()
            print(f"Created output directory: {output_dir}")
            
        # Set default file paths ONLY if sampling_file is not provided
        if sampling_file is None:
            if output_dir is None:
                # This should not happen, but fallback to current directory
                output_dir = "."
            sampling_file = os.path.join(output_dir, "loxcode_census_at_sampling.csv")

        # Define generation counter parameters
        h = 1.0  # Survival rate
        
        # Define division time parameters
        time_factor = 0.95
        m = 12 * time_factor
        s = 2.54 * time_factor
        sigma = np.log(1 + (s/m)**2)
        mu = np.log(m) - 0.5 * sigma
        
        # Initialize random number generator
        np.random.seed(int(py_time.time()))
        
        # Calculate probability distribution
        diff_probs = [[] for _ in range(self.transition_matrix.shape[0])]
        for i in range(self.transition_matrix.shape[0]):
            for j in range(self.transition_matrix.shape[1]):
                diff_probs[i].append(self.transition_matrix[i, j])
        
        # Initialize population
        # Store cells in dictionary with time as key and (barcode, fate path) as value
        population = {}
        
        # Set initial time and next division time
        time = 40
        next_split = 40 + 0.5
        
        # Add initial cells
        div_time = lambda: time + np.random.lognormal(mu, np.sqrt(sigma))
        
        c1 = deque([1, 1, -1, 0])
        c2 = deque([0, 1, -1, 0])
        c3 = deque([1, 0, -1, 0])
        c4 = deque([0, 0, -1, 0])
        
        population[div_time()] = (self.lox_original.copy(), c1)
        population[div_time()] = (self.lox_original.copy(), c2)
        population[div_time()] = (self.lox_original.copy(), c3)
        population[div_time()] = (self.lox_original.copy(), c4)
        
        # Create output files - only create files that are specified
        f_pedigree = open(pedigree_file, "w") if pedigree_file else None
        f_barcoding = open(barcoding_file, "w") if barcoding_file else None
        f_collection = open(sampling_file, "w")
        
        # Write headers
        if f_barcoding:
            f_barcoding.write("loxcode, binary pedigree, fate path\n")
        f_collection.write("loxcode, binary pedigree, fate path\n")
        
        # Create write buffers for I/O optimization
        pedigree_buffer = []
        barcoding_buffer = []
        collection_buffer = []
        buffer_limit = 10000  # Flush after this many lines
        
        # Main simulation loop
        codes = []
        round_num = 0
        
        print(f"Starting simulation, barcoding time: {t_barcoding} hours, collection time: {t_collection} hours")
        
        # Create progress markers before barcoding
        time_markers = [40, 60, 80, 100, t_barcoding-10, t_barcoding]
        if t_barcoding < t_collection:
            # Add additional time points after barcoding
            additional_markers = [t_barcoding + (t_collection-t_barcoding)/4,
                                t_barcoding + (t_collection-t_barcoding)/2,
                                t_barcoding + 3*(t_collection-t_barcoding)/4]
            time_markers.extend(additional_markers)
        
        next_marker_idx = 0
        
        while True:
            round_num += 1
            if round_num % 1000 == 0:
                print(f"Simulation round {round_num}, current time: {time:.2f} hours, population size: {len(population)}")
                
            if len(population) == 0:
                print("Population extinct, simulation ended")
                break
                
            # Get the next cell to divide
            next_time = min(population.keys())
            time = next_time
            loxcode, fate_path = population[next_time]
            del population[next_time]
            
            # Time progress report
            if next_marker_idx < len(time_markers) and time >= time_markers[next_marker_idx]:
                progress = min(100, int((time / t_collection) * 100))
                print(f"📊 Simulation progress: {progress}% - Time: {time:.1f} hours, population size: {len(population)}")
                next_marker_idx += 1
            
            # Cell division or death
            if np.random.random() < h:  # Cell divides
                # Record lineage - only if pedigree file is specified
                if f_pedigree:
                    pedigree_line = f"{time},"
                    for p in fate_path:
                        pedigree_line += f"{p} "
                    pedigree_line += "\n"
                    pedigree_buffer.append(pedigree_line)
                    
                    # Flush buffer if it gets too large
                    if len(pedigree_buffer) > buffer_limit:
                        f_pedigree.writelines(pedigree_buffer)
                        pedigree_buffer = []
                
                # Create daughter cell fate paths
                ped0 = deque(fate_path)
                ped1 = deque(fate_path)
                ped0.appendleft(0)
                ped1.appendleft(1)
                
                # Determine division time function
                if fate_path[-1] != 16:  # Non-blood cells
                    div_time_func = div_time
                else:  # Blood cells divide more slowly
                    div_time_func = lambda: time + np.random.lognormal(1.3*mu, 1.3*np.sqrt(sigma))
                
                # Add two daughter cells
                population[div_time_func()] = (loxcode.copy(), ped0)
                population[div_time_func()] = (loxcode.copy(), ped1)
            
            # Barcode marking
            if time > t_barcoding and t_barcoding != -1:
                N = [len(population)]
                P = [4]  # Average 4 recombination events per cell
                cassette_size = 13
                codes = self.generate_codes(N, P, cassette_size)
                
                # Assign barcodes to each cell
                counter = 0
                for cell_time in sorted(population.keys()):
                    loxcode, fate = population[cell_time]
                    population[cell_time] = (codes[counter].copy(), fate)
                    counter += 1
                
                # Census at barcoding time - only if barcoding file is specified
                if f_barcoding:
                    for cell_time, (loxcode, fate_path) in population.items():
                        barcoding_line = " ".join(map(str, loxcode)) + ","
                        for p in fate_path:
                            if p == -1:
                                barcoding_line += ","
                            else:
                                barcoding_line += f"{p} "
                        barcoding_line += "\n"
                        barcoding_buffer.append(barcoding_line)
                    
                    # Write all barcoding data at once
                    f_barcoding.writelines(barcoding_buffer)
                    barcoding_buffer = []
                
                # Set to -1 to avoid repeated marking
                t_barcoding = -1
                print(f"Barcoding completed, current time: {time:.2f} hours, population size: {len(population)}")
            
            # Fate determination - OPTIMIZED WITH BATCH PROCESSING
            if time > next_split:
                # Collect all cells that need to be processed
                cell_times = []
                current_states = []
                cell_data = []  # To keep track of which cells are being processed
                
                for cell_time in list(population.keys()):
                    loxcode, fate_path = population[cell_time]
                    current_state = fate_path[-1]
                    
                    if current_state < 16:  # Not already terminal cell type
                        cell_times.append(cell_time)
                        current_states.append(current_state)
                        cell_data.append((loxcode, fate_path))
                
                if len(current_states) > 0:
                    # Convert lists to numpy arrays for the JIT function
                    current_states_array = np.array(current_states)
                    
                    # Call the JIT-optimized function for batch processing
                    # The function will determine which cells change fate based on probabilities
                    transition_matrix_array = np.array(self.transition_matrix)
                    new_states = calculate_fate_decisions(
                        current_states_array, 
                        self.split_times,
                        time, 
                        self.diff_speed,
                        transition_matrix_array
                    )
                    
                    # Update the cells that changed state
                    for i, (cell_time, new_state) in enumerate(zip(cell_times, new_states)):
                        if new_state != current_states[i]:
                            loxcode, fate_path = cell_data[i]
                            fate_path.append(int(new_state))  # Convert from numpy type to Python int
                            population[cell_time] = (loxcode, fate_path)
                
                next_split = time + 1
            
            # End condition
            if time > t_collection:
                print(f"Sampling time reached, simulation ended, final population size: {len(population)}")
                
                # Record population state at sampling time - buffered write
                for cell_time, (loxcode, fate_path) in population.items():
                    collection_line = " ".join(map(str, loxcode)) + ","
                    for p in fate_path:
                        if p == -1:
                            collection_line += ","
                        else:
                            collection_line += f"{p} "
                    collection_line += "\n"
                    collection_buffer.append(collection_line)
                
                # Write all collection data at once
                f_collection.writelines(collection_buffer)
                collection_buffer = []
                
                break
        
        # Write any remaining data in buffers
        if pedigree_buffer and f_pedigree:
            f_pedigree.writelines(pedigree_buffer)
        if barcoding_buffer and f_barcoding:
            f_barcoding.writelines(barcoding_buffer)
        if collection_buffer:
            f_collection.writelines(collection_buffer)
        
        # Close files
        if f_pedigree:
            f_pedigree.close()
        if f_barcoding:
            f_barcoding.close()
        f_collection.close()
        
        print("Simulation completed, results written to files")

    def sim_multiple(self, num_embryos=3, t_barcoding=132, t_collection=228, n_processes=None, output_dir=None):
        """
        Use multiprocessing to simulate multiple embryo developments in parallel
        
        Parameters:
        num_embryos: Number of embryos to simulate
        t_barcoding: Barcoding time (hours)
        t_collection: Sampling time (hours)
        n_processes: Number of processes to use, None means use all available CPUs
        output_dir: Directory to save output files
        """
        # If no output directory specified, create one based on date
        if output_dir is None:
            output_dir = create_date_directory()
            print(f"Created output directory: {output_dir}")

        # If process count not specified, use CPU count
        if n_processes is None:
            n_processes = os.cpu_count()
        
        # Limit processes to not exceed embryo count
        n_processes = min(n_processes, num_embryos)
        
        print(f"Using {n_processes} processes to simulate {num_embryos} embryos in parallel")
        
        # Create parameters for each embryo simulation
        params_list = []
        for i in range(num_embryos):
            params = (i, num_embryos, t_barcoding, t_collection, 
                    self.times, self.fluxes, self.diff_speed, self.lox_original, output_dir)
            params_list.append(params)
        
        # Use process pool to execute simulations in parallel
        embryo_files = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all tasks with the standalone function
            futures = [executor.submit(_simulate_single_embryo, params) for params in params_list]
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    file = future.result()
                    embryo_files.append(file)
                    print(f"Embryo #{i+1} simulation completed")
                except Exception as e:
                    print(f"Embryo #{i+1} simulation failed: {e}")
        
        return embryo_files
    
    def analyze_results(self, sampling_files=None, output_prefix=None):
        """
        Analyze sampling results, generate tissue-barcode matrix and calculate inter-tissue similarity
        using fate coupling on concatenated matrix from all embryos
        
        Parameters:
        sampling_files: List of sampling data file names, if None use default file
        output_prefix: Output file prefix, if None use date directory + 'results'
        """
        import pandas as pd
        import numpy as np
        import os
        
        # Try to import the required packages, provide helpful error if not available
        try:
            import scanpy as sc
            import anndata as ad
            import cospar as cs
        except ImportError:
            print("Error: This analysis requires scanpy, anndata, and cospar packages.")
            print("Please install them with: pip install scanpy anndata cospar")
            return None
        
        # Handle output directory logic 
        if output_prefix is None:
            # Only create directory if no sampling files provided
            if sampling_files is None:
                output_dir = create_date_directory()
                output_prefix = os.path.join(output_dir, "results")
                sampling_files = [os.path.join(output_dir, "loxcode_census_at_sampling.csv")]
                print(f"Created analysis output directory: {output_dir}")
            else:
                # Use the directory of the first sampling file
                output_dir = os.path.dirname(sampling_files[0])
                output_prefix = os.path.join(output_dir, "results")
        
        # Error check if no sampling files provided
        if sampling_files is None:
            print("Error: No sampling files provided")
            return None
            
        print("Starting analysis of sampling results...")   
                
        # Process data for each embryo
        all_embryo_data = []
        for file_idx, sampling_file in enumerate(tqdm(sampling_files, desc="Reading embryo data")):
            print(f"Processing embryo #{file_idx+1} data: {sampling_file}")
            embryo_data = []
            
            # Read sampling data
            with open(sampling_file, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue
                        
                    loxcode = tuple(map(int, parts[0].strip().split()))
                    
                    # Extract the last state (terminal cell type) from fate path
                    fate_parts = parts[-1].strip().split()
                    if len(fate_parts) > 0:
                        tissue_type = int(fate_parts[-1])
                        # Only focus on terminal cell types 16-32
                        if tissue_type >= 16:
                            embryo_data.append((loxcode, tissue_type))
            
            all_embryo_data.append(embryo_data)
        
        # Define tissue type name mapping
        tissue_names = {
            16: "blood",
            17: "L brain I", 18: "R brain I",
            19: "L brain III", 20: "R brain III",
            21: "L gonad", 22: "R gonad",
            23: "L kidney", 24: "R kidney",
            25: "L foot", 26: "L leg",
            27: "R foot", 28: "R leg",
            29: "L hand", 30: "L arm",
            31: "R hand", 32: "R arm"
        }
        
        # Create tissue-barcode count matrix for each embryo and save only first 3
        embryo_matrices = []
        all_barcode_matrices = []  # Store all matrices for concatenation
        individual_coupling_matrices = []  # Store individual fate coupling results
        
        for embryo_idx, embryo_data in enumerate(all_embryo_data):
            # Create tissue-barcode count matrix
            tissue_barcode_counts = {}
            for tissue_id in range(16, 33):
                tissue_barcode_counts[tissue_id] = collections.Counter()
            
            for barcode, tissue_id in embryo_data:
                tissue_barcode_counts[tissue_id][barcode] += 1
            
            # Get all unique barcodes
            all_barcodes = set()
            for tissue_id, counter in tissue_barcode_counts.items():
                all_barcodes.update(counter.keys())
            all_barcodes = list(all_barcodes)
            
            # Create tissue-barcode matrix
            matrix_data = []
            tissue_ids = sorted(tissue_barcode_counts.keys())
            
            for tissue_id in tissue_ids:
                row = [tissue_barcode_counts[tissue_id][barcode] for barcode in all_barcodes]
                matrix_data.append(row)
            
            # Create DataFrame with embryo-specific barcode names
            barcode_labels = [f"embryo{embryo_idx+1}_{str(b)}" for b in all_barcodes]
            tissue_labels = [tissue_names[tid] for tid in tissue_ids]
            barcode_matrix = pd.DataFrame(matrix_data, index=tissue_labels, columns=barcode_labels)
            
            # Save individual tissue-barcode matrix only for first 3 embryos
            if embryo_idx < 3:
                barcode_matrix.to_csv(f"{output_prefix}_embryo{embryo_idx+1}_tissue_barcode_matrix.csv")
                print(f"Individual matrix saved for embryo #{embryo_idx+1}")
                
                # ADDED: Calculate individual fate coupling for first 3 embryos
                print(f"Calculating fate coupling for embryo #{embryo_idx+1}...")
                
                # Check if matrix has enough data for analysis
                if len(all_barcodes) > 0 and barcode_matrix.sum().sum() > 0:
                    try:
                        # Create AnnData object for individual embryo
                        adata_individual = ad.AnnData(barcode_matrix)
                        adata_individual.obs['state_info'] = barcode_matrix.index
                        adata_individual.obsm["X_clone"] = barcode_matrix.values
                        adata_individual.obs['time_info'] = f'embryo_{embryo_idx+1}'
                        adata_individual.uns["data_des"] = [f"LoxCode embryo {embryo_idx+1}"]
                        
                        # Calculate fate coupling for individual embryo
                        cs.tl.fate_coupling(adata_individual, source="X_clone")
                        
                        # Extract individual coupling matrix
                        individual_coupling = pd.DataFrame(
                            adata_individual.uns['fate_coupling_X_clone']['X_coupling'],
                            index=adata_individual.uns['fate_coupling_X_clone']['fate_names'],
                            columns=adata_individual.uns['fate_coupling_X_clone']['fate_names']
                        )
                        
                        # Fix diagonal precision issues for individual matrix
                        print(f"Fixing diagonal values for embryo #{embryo_idx+1} correlation matrix...")
                        np.fill_diagonal(individual_coupling.values, 1.0)
                        individual_coupling = individual_coupling.round(10)
                        individual_coupling = (individual_coupling + individual_coupling.T) / 2
                        np.fill_diagonal(individual_coupling.values, 1.0)
                        
                        # Save individual fate coupling matrix
                        individual_coupling.to_csv(f"{output_prefix}_embryo{embryo_idx+1}_fate_coupling.csv")
                        print(f"Individual fate coupling matrix saved for embryo #{embryo_idx+1}")
                        
                        individual_coupling_matrices.append(individual_coupling)
                        
                    except Exception as e:
                        print(f"Warning: Could not calculate fate coupling for embryo #{embryo_idx+1}: {e}")
                        individual_coupling_matrices.append(None)
                else:
                    print(f"Warning: Insufficient data for fate coupling analysis in embryo #{embryo_idx+1}")
                    individual_coupling_matrices.append(None)
                
            # Store all matrices for concatenation
            all_barcode_matrices.append(barcode_matrix)
            
            # log(1+x) transformation for standard analysis
            log_barcode_matrix = np.log1p(barcode_matrix)
            embryo_matrices.append(log_barcode_matrix)
        
        # Concatenate all embryo matrices horizontally (column-wise)
        print("Concatenating all embryo matrices...")
        concatenated_matrix = pd.concat(all_barcode_matrices, axis=1)
        
        # Save concatenated matrix
        concatenated_matrix.to_csv(f"{output_prefix}_concatenated_tissue_barcode_matrix.csv")
        print(f"Concatenated matrix saved with shape: {concatenated_matrix.shape}")
        
        # Calculate fate coupling using COSPAR on concatenated matrix
        print("Calculating fate coupling matrix on concatenated data...")
        
        # Create AnnData object directly from concatenated matrix (tissues as rows, barcodes as columns)
        adata = ad.AnnData(concatenated_matrix)  
        adata.obs['state_info'] = concatenated_matrix.index  
        adata.obsm["X_clone"] = concatenated_matrix.values 
        adata.obs['time_info'] = 'all_embryos'
        adata.uns["data_des"] = ["LoxCode all embryos combined"]

        # Calculate fate coupling
        cs.tl.fate_coupling(adata, source="X_clone")
            
        # Check the sparsity of the concatenated matrix
        total_cells = np.sum(concatenated_matrix.values)
        non_zero = np.count_nonzero(concatenated_matrix.values)
        print(f"  Concatenated matrix: {concatenated_matrix.shape}, total cells: {total_cells}, non-zero elements: {non_zero}")

        # Extract tissue-level coupling matrix directly
        coupling_matrix = pd.DataFrame(
            adata.uns['fate_coupling_X_clone']['X_coupling'],
            index=adata.uns['fate_coupling_X_clone']['fate_names'],
            columns=adata.uns['fate_coupling_X_clone']['fate_names']
        )

        # Save the final MCSA matrix
        coupling_matrix.to_csv(f"{output_prefix}_mcsa_matrix.csv")
        print(f"Final MCSA matrix saved based on concatenated data")
        
        return {
            'embryo_matrices': embryo_matrices[:3],  
            'concatenated_matrix': concatenated_matrix,
            'mcsa_matrix': coupling_matrix,
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate embryo development and LoxCode barcoding process')
    parser.add_argument('--barcoding', type=float, default=132, help='Barcoding time (hours)')
    parser.add_argument('--collection', type=float, default=228, help='Sampling time (hours)') 
    parser.add_argument('--num_embryos', '-n', type=int, default=100, help='Number of embryos to simulate')
    parser.add_argument('--processes', '-p', type=int, default=None, help='Number of processes to use, default uses all available CPUs')
    parser.add_argument('--analyze', action='store_true', help='Analyze results and generate similarity matrix')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_date_directory()
    print(f"Saving results to directory: {output_dir}")
    
    model = LoxCodeEmbryoModel()
    
    # Simulate multiple embryos
    if args.num_embryos > 1:
        sampling_files = model.sim_multiple(args.num_embryos, args.barcoding, 
                                          args.collection, args.processes, output_dir)
        if args.analyze:
            model.analyze_results(sampling_files, os.path.join(output_dir, "results"))
    else:
        # Keep original single embryo simulation behavior
        model.sim_single(args.barcoding, args.collection, output_dir=output_dir)
        if args.analyze:
            model.analyze_results(output_prefix=os.path.join(output_dir, "results"))

if __name__ == "__main__":
    # For cross-platform compatibility with multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()