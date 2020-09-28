import os, re, datetime, math, h5py
import numpy as np

class Tile:
    def __init__(self, hdf5path):
        self.granule_id = os.path.splitext(os.path.basename(hdf5path))[0]
        self.observation_date = datetime.datetime(int(self.granule_id[7:11]), int(self.granule_id[11:13]), int(self.granule_id[13:15]), tzinfo=datetime.timezone.utc)
        self.nodata_value = np.NaN

        self.vtile = int(self.granule_id[21:23])
        self.htile = int(self.granule_id[23:25])
        self.product = self.granule_id[31:35].replace('_', '')
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
            self.qa_flags = f['Image_data']['QA_flag'][:]
            self.qa_description = f['Image_data']['QA_flag'].attrs['Data_description'][0].decode()
            self.grid_interval_num = f['Image_data'].attrs['Grid_interval'][0]
            
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
            self.obs_time = obs_time * 3600.0 + np.ones((self.lines,self.pixels), dtype="float64") * self.observation_date.timestamp()

    def is_good(self):
        return self.status == 'Good'

    def get_point(self, line, pixel):
        value = self.data[line, pixel]     
        if np.isnan(value):
            raise Exception('error value.')

        obs_time = self.obs_time[line, pixel]
        if np.isnan(obs_time):
            raise Exception('error obs_time.')
        observation_datetime = datetime.datetime.fromtimestamp(obs_time, datetime.timezone.utc)
        
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
        
        qa_flag = int(self.qa_flags[line, pixel])
        qa_flags = [i for i, qa in enumerate(reversed(format(qa_flag, 'b'))) if int(qa) == 1]

        return {
            'location': [lon, lat],
            'value': value,
            'observation_datetime': observation_datetime,
            'qa_flags': qa_flags
        }