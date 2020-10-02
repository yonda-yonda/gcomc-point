import os, datetime, math, h5py
import numpy as np

class Scene:
    # support level: L2
    # support file: SSTD, IWPR, NWLR
    def __init__(self, hdf5path, product = None):
        self.granule_id = os.path.splitext(os.path.basename(hdf5path))[0]
        self.observation_date = datetime.datetime(int(self.granule_id[7:11]), int(self.granule_id[11:13]), int(self.granule_id[13:15]), tzinfo=datetime.timezone.utc)
        self.nodata_value = np.NaN

        self.path = int(self.granule_id[20:23])
        self.scene = int(self.granule_id[23:25])
        
        self.product = self.granule_id[31:35].replace('_', '') if product == None else product
        self.resolution = self.granule_id[35]

        with h5py.File(hdf5path, 'r') as f:
            self.status = f['Processing_attributes'].attrs['Processing_result'][0].decode() 
            processing_ut = f['Processing_attributes'].attrs['Processing_UT'][0].decode()
            self.processed = datetime.datetime.strptime(processing_ut + '+0000', '%Y%m%d %H:%M:%S%z')
            self.algorithm_version = f['Global_attributes'].attrs['Algorithm_version'][0].decode()
            start_time = f['Global_attributes'].attrs['Scene_start_time'][0].decode()
            self.start_time = datetime.datetime.strptime(start_time + '+0000', '%Y%m%d %H:%M:%S.%f%z')
            end_time = f['Global_attributes'].attrs['Scene_end_time'][0].decode()
            self.end_time = datetime.datetime.strptime(end_time + '+0000', '%Y%m%d %H:%M:%S.%f%z')

            self.lower_left = [float(f['Geometry_data'].attrs['Lower_left_longitude'][0]), float(f['Geometry_data'].attrs['Lower_left_latitude'][0])]
            self.lower_right = [float(f['Geometry_data'].attrs['Lower_right_longitude'][0]), float(f['Geometry_data'].attrs['Lower_right_latitude'][0])]
            self.upper_left = [float(f['Geometry_data'].attrs['Upper_left_longitude'][0]), float(f['Geometry_data'].attrs['Upper_left_latitude'][0])]
            self.upper_right = [float(f['Geometry_data'].attrs['Upper_right_longitude'][0]), float(f['Geometry_data'].attrs['Upper_right_latitude'][0])]

            data = f['Image_data'][self.product][:].astype(np.float64)
            slope = f['Image_data'][self.product].attrs['Slope'][0]
            offset = f['Image_data'][self.product].attrs['Offset'][0]
            error_dn = f['Image_data'][self.product].attrs['Error_DN'][0]
            min_valid_dn = f['Image_data'][self.product].attrs['Minimum_valid_DN'][0]
            max_valid_dn = f['Image_data'][self.product].attrs['Maximum_valid_DN'][0]
            self.unit = f['Image_data'][self.product].attrs['Unit'][0].decode()

            with np.errstate(invalid='ignore'):
                data[data == error_dn] = self.nodata_value
                data[(data < min_valid_dn) | (data > max_valid_dn)] = self.nodata_value
            self.data = slope * data + offset

            self.lines = int(f['Image_data'].attrs['Number_of_lines'][0])
            self.pixels = int(f['Image_data'].attrs['Number_of_pixels'][0])
            self.grid_interval_num = f['Image_data'].attrs['Grid_interval'][0]

            try:
                self.qa_flags = f['Image_data']['QA_flag'][:]
                self.qa_description = f['Image_data']['QA_flag'].attrs['Data_description'][0].decode()
            except:
                self.qa_flags = None
                self.qa_description = None
         
            try:
                obs_time = f['Geometry_data']['Obs_time'][:].astype(np.float64)
                obs_time_resampling = f['Geometry_data']['Obs_time'].attrs['Resampling_interval'][0]
                obs_time_offset = f['Geometry_data']['Obs_time'].attrs['Offset'][0]
                obs_time_slope = f['Geometry_data']['Obs_time'].attrs['Slope'][0]
                obs_time_error_dn = f['Geometry_data']['Obs_time'].attrs['Error_DN'][0]
                obs_time_min_valid_dn = f['Geometry_data']['Obs_time'].attrs['Minimum_valid_DN'][0]
                obs_time_max_valid_dn = f['Geometry_data']['Obs_time'].attrs['Maximum_valid_DN'][0]

                with np.errstate(invalid='ignore'):    
                    obs_time[obs_time == obs_time_error_dn] = self.nodata_value
                    obs_time[(obs_time < obs_time_min_valid_dn) | (obs_time > obs_time_max_valid_dn)] = self.nodata_value
                    obs_time = self._interp2d_biliner(obs_time, obs_time_resampling)
                    obs_time = obs_time_slope * obs_time + obs_time_offset

                self.obs_time = obs_time * 3600.0 + np.ones((self.lines, self.pixels), dtype="float64") * self.observation_date.timestamp()
            except:
                self.obs_time = None
                
            latitudes = f['Geometry_data']['Latitude'][:].astype(np.float64)
            latitudes_error_value = f['Geometry_data']['Latitude'].attrs['Error_value'][0]
            latitudes_min_valid_dn = f['Geometry_data']['Latitude'].attrs['Minimum_valid_value'][0]
            latitudes_max_valid_dn = f['Geometry_data']['Latitude'].attrs['Maximum_valid_value'][0]
            latitudes[latitudes == latitudes_error_value] = self.nodata_value
            latitudes[(latitudes < latitudes_min_valid_dn) | (latitudes > latitudes_max_valid_dn)] = self.nodata_value
            longitudes = f['Geometry_data']['Longitude'][:].astype(np.float64)
            longitudes_error_value = f['Geometry_data']['Longitude'].attrs['Error_value'][0]
            longitudes_min_valid_dn = f['Geometry_data']['Longitude'].attrs['Minimum_valid_value'][0]
            longitudes_max_valid_dn = f['Geometry_data']['Longitude'].attrs['Maximum_valid_value'][0]
            longitudes[longitudes == longitudes_error_value] = self.nodata_value
            longitudes[(longitudes < longitudes_min_valid_dn) | (longitudes > longitudes_max_valid_dn)] = self.nodata_value
            
            latitude_resampling = f['Geometry_data']['Latitude'].attrs['Resampling_interval'][0]
            longitude_resampling = f['Geometry_data']['Longitude'].attrs['Resampling_interval'][0]
            
            self.latitude = self._interp2d_biliner_coordinates(latitudes, latitude_resampling)
            self.longitude = self._interp2d_biliner_coordinates(longitudes, longitude_resampling, True)
            
    def _interp2d_biliner_coordinates(self, data, interval, is_lon=False):
        if is_lon is True:
            max_diff = np.nanmax(np.abs(data[:, :-1] - data[:,1:]))
            if max_diff > 180:
                data[data < 0] = 360 + data[data < 0]
        
        ret = self._interp2d_biliner(data, interval)
                    
        if is_lon == True:
            ret[ret > 180] = ret[ret > 180] - 360
        return ret
    
    def _interp2d_biliner(self, data, interval):
        # respect for https://github.com/K0gata/SGLI_Python_Open_Tool            
        data = np.concatenate((data, data[-1].reshape(1, -1)), axis=0)
        data = np.concatenate(
            (data, data[:, -1].reshape(-1, 1)), axis=1)
        ratio_horizontal = np.tile(np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32),
                                (data.shape[0]*interval, data.shape[1] - 1))
        ratio_vertical = np.tile(np.linspace(0, (interval - 1) / interval, interval, dtype=np.float64).reshape(-1, 1),
                                (data.shape[0] - 1, (data.shape[1] - 1)*interval))
        repeat_data = np.repeat(data, interval, axis=0)
        repeat_data = np.repeat(repeat_data, interval, axis=1)
        horizontal_interp = (1. - ratio_horizontal) * \
            repeat_data[:, :-interval] + \
            ratio_horizontal * repeat_data[:, interval:]
        ret = (1. - ratio_vertical) * \
            horizontal_interp[:-interval, :] + \
            ratio_vertical * horizontal_interp[interval:, :]

        return ret[:self.lines, :self.pixels]
            
    def is_good(self):
        return self.status == 'Good'

    def get_point(self, line, pixel):
        value = self.data[line, pixel]     
        if np.isnan(value):
            raise Exception('error value.')

        try:
            obs_time = self.obs_time[line, pixel]
            if np.isnan(obs_time):
                raise Exception
            observation_datetime = datetime.datetime.fromtimestamp(obs_time, datetime.timezone.utc)
        except:
            observation_datetime = None
        
       
        lat = self.latitude[line, pixel]
        lon = self.longitude[line, pixel]
        
        try:
            qa_flag = int(self.qa_flags[line, pixel])
            qa_flags = [i for i, qa in enumerate(reversed(format(qa_flag, 'b'))) if int(qa) == 1]
        except:
            qa_flags = None

        return {
            'location': [lon, lat],
            'value': value,
            'observation_datetime': observation_datetime,
            'qa_flags': qa_flags
        }  


    def _interp2d_biliner(self, data, interval):
        data = np.concatenate((data, data[-1].reshape(1, -1)), axis=0)
        data = np.concatenate(
            (data, data[:, -1].reshape(-1, 1)), axis=1)
        ratio_horizontal = np.tile(np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32),
                                (data.shape[0]*interval, data.shape[1] - 1))
        ratio_vertical = np.tile(np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32).reshape(-1, 1),
                                (data.shape[0] - 1, (data.shape[1] - 1)*interval))
        repeat_data = np.repeat(data, interval, axis=0)
        repeat_data = np.repeat(repeat_data, interval, axis=1)
        horizontal_interp = (1. - ratio_horizontal) * \
            repeat_data[:, :-interval] + \
            ratio_horizontal * repeat_data[:, interval:]
        ret = (1. - ratio_vertical) * \
            horizontal_interp[:-interval, :] + \
            ratio_vertical * horizontal_interp[interval:,:]

        return ret

class Tile:
    # support level: L2
    # support file: LST, AGB, LAI, VGI, CLFG, CLPR, ARNP, ARPLK
    def __init__(self, hdf5path, product = None):
        self.granule_id = os.path.splitext(os.path.basename(hdf5path))[0]
        self.observation_date = datetime.datetime(int(self.granule_id[7:11]), int(self.granule_id[11:13]), int(self.granule_id[13:15]), tzinfo=datetime.timezone.utc)
        self.nodata_value = np.NaN

        self.vtile = int(self.granule_id[21:23])
        self.htile = int(self.granule_id[23:25])
        self.product = self.granule_id[31:35].replace('_', '') if product == None else product
        self.resolution = self.granule_id[35]

        with h5py.File(hdf5path, 'r') as f:
            self.status = f['Processing_attributes'].attrs['Processing_result'][0].decode() 
            processing_ut = f['Processing_attributes'].attrs['Processing_UT'][0].decode()
            self.processed = datetime.datetime.strptime(processing_ut + '+0000', '%Y%m%d %H:%M:%S%z')
            self.algorithm_version = f['Global_attributes'].attrs['Algorithm_version'][0].decode()
            start_time = f['Global_attributes'].attrs['Image_start_time'][0].decode()
            self.start_time = datetime.datetime.strptime(start_time + '+0000', '%Y%m%d %H:%M:%S.%f%z')
            end_time = f['Global_attributes'].attrs['Image_end_time'][0].decode()
            self.end_time = datetime.datetime.strptime(end_time + '+0000', '%Y%m%d %H:%M:%S.%f%z')

            self.lower_left = [float(f['Geometry_data'].attrs['Lower_left_longitude'][0]), float(f['Geometry_data'].attrs['Lower_left_latitude'][0])]
            self.lower_right = [float(f['Geometry_data'].attrs['Lower_right_longitude'][0]), float(f['Geometry_data'].attrs['Lower_right_latitude'][0])]
            self.upper_left = [float(f['Geometry_data'].attrs['Upper_left_longitude'][0]), float(f['Geometry_data'].attrs['Upper_left_latitude'][0])]
            self.upper_right = [float(f['Geometry_data'].attrs['Upper_right_longitude'][0]), float(f['Geometry_data'].attrs['Upper_right_latitude'][0])]

            data = f['Image_data'][self.product][:].astype(np.float64)
            slope = f['Image_data'][self.product].attrs['Slope'][0]
            offset = f['Image_data'][self.product].attrs['Offset'][0]
            error_dn = f['Image_data'][self.product].attrs['Error_DN'][0]
            min_valid_dn = f['Image_data'][self.product].attrs['Minimum_valid_DN'][0]
            max_valid_dn = f['Image_data'][self.product].attrs['Maximum_valid_DN'][0]
            self.unit = f['Image_data'][self.product].attrs['Unit'][0].decode()

            with np.errstate(invalid='ignore'):
                data[data == error_dn] = self.nodata_value
                data[(data < min_valid_dn) | (data > max_valid_dn)] = self.nodata_value
            self.data = slope * data + offset

            self.lines = int(f['Image_data'].attrs['Number_of_lines'][0])
            self.pixels = int(f['Image_data'].attrs['Number_of_pixels'][0])
            self.grid_interval_num = f['Image_data'].attrs['Grid_interval'][0]

            try:
                self.qa_flags = f['Image_data']['QA_flag'][:]
                self.qa_description = f['Image_data']['QA_flag'].attrs['Data_description'][0].decode()
            except:
                self.qa_flags = None
                self.qa_description = None
            
            try:
                obs_time = f['Geometry_data']['Obs_time'][:].astype(np.float64)
                obs_time_offset = f['Geometry_data']['Obs_time'].attrs['Offset'][0]
                obs_time_slope = f['Geometry_data']['Obs_time'].attrs['Slope'][0]
                obs_time_error_dn = f['Geometry_data']['Obs_time'].attrs['Error_DN'][0]
                obs_time_min_valid_dn = f['Geometry_data']['Obs_time'].attrs['Minimum_valid_DN'][0]
                obs_time_max_valid_dn = f['Geometry_data']['Obs_time'].attrs['Maximum_valid_DN'][0]

                with np.errstate(invalid='ignore'):    
                    obs_time[obs_time == obs_time_error_dn] = self.nodata_value
                    obs_time[(obs_time < obs_time_min_valid_dn) | (obs_time > obs_time_max_valid_dn)] = self.nodata_value
                    obs_time = obs_time_slope * obs_time + obs_time_offset
                self.obs_time = obs_time * 3600.0 + np.ones((self.lines, self.pixels), dtype="float64") * self.observation_date.timestamp()
            except:
                self.obs_time = None

    def is_good(self):
        return self.status == 'Good'

    def get_point(self, line, pixel):
        value = self.data[line, pixel]     
        if np.isnan(value):
            raise Exception('error value.')

        try:
            obs_time = self.obs_time[line, pixel]
            if np.isnan(obs_time):
                raise Exception
            observation_datetime = datetime.datetime.fromtimestamp(obs_time, datetime.timezone.utc)
        except:
            observation_datetime = None
        
        vtile_num = 18
        v_pixel = int(self.lines * vtile_num)
        d = 180 / v_pixel
        NL = v_pixel  # 180 / d
        NP0 = 2 * NL
    
        lin_total = line + (self.vtile * self.lines)
        col_total = pixel + (self.htile * self.pixels)
        lat = 90 - (lin_total + 0.5) * d
        NPi = int(round(NP0 * math.cos(math.radians(lat)), 0))
        lon = 360 / NPi * (col_total - NP0 / 2 + 0.5)
        
        try:
            qa_flag = int(self.qa_flags[line, pixel])
            qa_flags = [i for i, qa in enumerate(reversed(format(qa_flag, 'b'))) if int(qa) == 1]
        except:
            qa_flags = None

        return {
            'location': [lon, lat],
            'value': value,
            'observation_datetime': observation_datetime,
            'qa_flags': qa_flags
        }