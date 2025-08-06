import math
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

class make_dataset_maps():
    def __init__(self, maps, params, 
                 rotations=True, 
                 smoothing=False, 
                 k_smooth=None, k_min=2., k_max=45., normalize_k=False, 
                 BoxSize=25., 
                 linear=False, 
                 log_scale=False, 
                 standardize=False, maps_mean=None, maps_std=None):
        """
        Dataset class for storing and processing maps with Gaussian smoothings.
        
        Args:
            maps (torch.tensor): input 2D maps
            params (torch.tensor): parameters corresponding to the maps
            rotations (bool, optional): apply random flips and rotations to the maps
            smoothing (bool, optional): apply Gaussian smoothing to maps
            k_smooth (float, optional): wavenumber (in h/Mpc) corresponding to a specific smoothing scale for Gaussian smoothing. If None, a random smoothing scale is chosen. 
            k_min (float, optional): minimum wavenumber k (in h/Mpc) for the random sampling interval 
            k_max (float, optional): maximum wavenumber k (in h/Mpc) for the random sampling interval
            normalize_k (bool, optional): normalize wavenumber k w.r.t. to the maximum wavenumber k_max
            BoxSize (float, optional): length of simulation box (in Mpc/h)
            linear (bool, optional): if True, wavenumbers are sampled linearly. Otherwise, log-space sampling is applied.
            log_scale (bool, optional): output maps with log scaling
            standardize (bool, optional): output maps standardized with mean set to maps_mean and std set to maps_std. Note that standardization follows log-scaling (if applied).
            maps_mean (float, optional): mean value for standardizing the maps. 
            maps_std (float, optional): standard deviation for standardizing the maps. 
            
        """
        self.total_sims   = maps.shape[0]
        self.size_map     = maps.shape[-1]
        self.total_params = params.shape[-1]
        
        maps   = maps.reshape(-1, 1, self.size_map, self.size_map)
        params = params.reshape(-1, self.total_params)
        self.size      = maps.shape[0]
        self.x         = maps
        self.y         = params
        
        self.rotations = rotations
        self.smoothing = smoothing
        self.k_smooth  = k_smooth
        self.k_min     = k_min
        self.k_max     = k_max
        self.norm_k    = normalize_k
        
        self.BoxSize     = BoxSize
        self.linear      = linear
        
        self.log_scale   = log_scale
        self.standardize = standardize
        
        if standardize:
            if maps_mean is None:  raise ValueError("Mean value for standardization is missing.")
            if maps_std is None:  raise ValueError("Standard deviation for standardization is missing.")
                
        self.maps_mean   = maps_mean
        self.maps_std    = maps_std
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Returns a tuple of map at index idx, corresponding parameters, and smoothing scale (0 if no smoothing was applied).
        """
        maps  = self.x[idx]
        
        if self.rotations:
            # choose a rotation angle (0-0, 1-90, 2-180, 3-270)
            # and whether do a random flip or not
            rot  = np.random.randint(0,4)
            flip = np.random.randint(0,2)

            # rotate and flip the maps
            maps = torch.rot90(maps, k=rot, dims=[1,2])
            if flip==1:  maps = torch.flip(maps, dims=[1])

        # define smoothing wavevector k
        if self.smoothing is False:
            k   = np.array([0.]).reshape(1,)
        elif self.k_smooth is not None:
            k   = np.array([self.k_smooth])
        else:
            # randomly sample smoothing scale value
            rng = np.random.default_rng()
            if self.linear:
                k   = rng.uniform(low=self.k_min, high=self.k_max, size=(1,))
            else:
                k   = 10**rng.uniform(low=np.log10(self.k_min), high=np.log10(self.k_max), size=(1,)) 
        
        # (optional) apply smoothing
        if self.smoothing:
            R     = 1/k
            sigma = R/(self.BoxSize/self.size_map)
            maps  = torch.tensor(gaussian_filter(maps[0], sigma[0], truncate=5.0, mode='wrap').reshape(1, self.size_map, self.size_map))
        
        # (optional) scale and standardize maps
        if self.log_scale:
            maps = torch.log10(maps)     
        if self.standardize:
            maps   = (maps - self.maps_mean)/self.maps_std
        
        if self.norm_k:
            if self.linear:
                return maps, self.y[idx], torch.tensor(k/self.k_max)
            else:
                # if wavevectors are sampled in log-space, normalization is done in log-space as well
                return maps, self.y[idx], torch.tensor(np.log10(k)/np.log10(self.k_max))
        else:
            return maps, self.y[idx], torch.tensor(k)
        
def create_datasets_maps(maps, params, 
                         train_frac, valid_frac, test_frac, 
                         seed = None, 
                         rotations=True, 
                         smoothing=False, 
                         k_smooth=None, k_min=2., k_max=45., 
                         normalize_k=False,
                         BoxSize=25., 
                         linear=False, 
                         log_scale=False, 
                         standardize=False, maps_mean=None, maps_std=None,): 
    """Create training, validation, and test datasets for density maps with (optional) Gaussian smoothing.
    
    Args:
        maps (torch.tensor): input 2D maps
        params (torch.tensor): parameters corresponding to the maps
        train_frac (float): fraction of data reserved for training
        valid_frac (float): fraction of data reserved for validation
        test_frac (float): fraction of data reserved for testing
        seed (int, optional): random seed to generate train/validation/test split
        rotations (bool, optional): whether to apply random flips and rotations to the maps
        smoothing (bool, optional): whether to apply Gaussian smoothing to maps
        k_smooth (float, optional): wavenumber (in h/Mpc) corresponding to a specific smoothing scale for Gaussian smoothing. If None, a random smoothing scale is chosen. 
        k_min (float, optional): minimum wavenumber k (in h/Mpc) for the random sampling interval 
        k_max (float, optional): maximum wavenumber k (in h/Mpc) for the random sampling interval
        normalize_k (bool, optional): normalize wavenumber k w.r.t. to the maximum wavenumber k_max
        BoxSize (float, optional): length of simulation box (in Mpc/h)
        linear (bool, optional): if True, wavenumbers are sampled linearly. Otherwise, log-space sampling is applied.
        log_scale (bool, optional): output maps with log scaling
        standardize (bool, optional): output maps standardized with mean set to maps_mean and std set to maps_std. Note that standardization follows log-scaling (if applied).
        maps_mean (float, optional): mean value for standardizing the maps. 
        maps_std (float, optional): standard deviation for standardizing the maps. 
           
    
    Returns:
        Tuple: training, validation, and test datasets
    """
    assert maps.shape[0] == params.shape[0]
    assert math.isclose(train_frac + valid_frac + test_frac, 1.)
    
    dset_size = maps.shape[0]
    # if no mean and/or std provided for smoothing, calculate mean and std from the data
    if standardize:
        if maps_mean is None:
            if log_scale:  maps_mean = np.log10(maps).mean()
            else:  maps_mean = maps.mean()
        if maps_std is None:
            if log_scale:  maps_std = np.log10(maps).std()
            else:  maps_std = maps.std()
    
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
        # randomly shuffle the simulations 
        sim_numbers = np.arange(dset_size) 
        rng.shuffle(sim_numbers)

    # get indices of shuffled maps
    train_size, valid_size, test_size = int(train_frac*dset_size), int(valid_frac*dset_size), int(test_frac*dset_size)
    train_ind = sim_numbers[:train_size]
    valid_ind = sim_numbers[train_size:(train_size+valid_size)]
    test_ind  = sim_numbers[(train_size+valid_size):]
                
    maps_train, params_train = maps[train_ind], params[train_ind]
    maps_valid, params_valid = maps[valid_ind], params[valid_ind]
    maps_test, params_test   = maps[test_ind],  params[test_ind]
    

    # make train/valid/test datasets 
    train_dset = make_dataset_maps(maps_train, params_train, 
                              rotations=rotations, 
                              smoothing=smoothing, 
                              k_smooth=k_smooth, k_min=k_min, k_max=k_max, normalize_k=normalize_k,
                              BoxSize=BoxSize, 
                              linear=linear, 
                              log_scale=log_scale, 
                              standardize=standardize, maps_mean=maps_mean, maps_std=maps_std)
    
    valid_dset = make_dataset_maps(maps_valid, params_valid, 
                              rotations=rotations, 
                              smoothing=smoothing, 
                              k_smooth=k_smooth, k_min=k_min, k_max=k_max, normalize_k=normalize_k,
                              BoxSize=BoxSize, 
                              linear=linear, 
                              log_scale=log_scale, 
                              standardize=standardize, maps_mean=maps_mean, maps_std=maps_std)
    
    test_dset  = make_dataset_maps(maps_test, params_test,  
                              rotations=rotations, 
                              smoothing=smoothing, 
                              k_smooth=k_smooth, k_min=k_min, k_max=k_max, normalize_k=normalize_k,
                              BoxSize=BoxSize, 
                              linear=linear, 
                              log_scale=log_scale, 
                              standardize=standardize, maps_mean=maps_mean, maps_std=maps_std)
    
    return train_dset, valid_dset, test_dset
