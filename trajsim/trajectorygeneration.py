"""Trajectory Generation Module
"""

# Modules
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import skmob
from skmob.models.epr import Ditras
from skmob.models.markov_diary_generator import MarkovDiaryGenerator
from skmob.preprocessing import filtering, compression, detection, clustering

# Parameters

# Methods

# Classes
## Base
class TrajectoryGenerationHourlyDITRAS:
    """Hourly Trajectory Generation using DITRAS.
    """
    # Constructor
    def __init__( self, path_markov_diary_generator_matrix, path_tessellation ):
        # Parameters
        self.path_markov_diary_generator_matrix = path_markov_diary_generator_matrix
        self.path_tessellation = path_tessellation
        # Initialize
        self._initialize()
        # Return
        return
    # Initialize
    def _initialize( self ):
        ########################################################################
        # Markov Diary Generator
        self.smdg = pd.read_csv( self.path_markov_diary_generator_matrix, header = None ).values
        self.mdg = MarkovDiaryGenerator()
        self.mdg._create_empty_markov_chain()
        n_rows, n_cols = self.smdg.shape
        for i in range(n_rows):
            for j in range(n_cols):
                # Skip Nans
                if( np.isnan( self.smdg[i,j] ) or self.smdg[i,j] == 0 ):
                    continue
                # Extract Info
                ## Location
                loc = 1 if i < 24 else 0
                loc_next = 1 if j < 24 else 0
                ## Hour
                hour = i % 24
                hour_next = j % 24
                # Update
                self.mdg.markov_chain_[(hour,loc)][(hour_next,loc_next)] = self.smdg[i,j]
        self.mdg._normalize_markov_chain()
        self.ditras = Ditras( self.mdg )
        ########################################################################
        # Tessellation Load
        self.df_tehran_pop = pd.read_csv( self.path_tessellation )
        self.df_tehran_pop = self.df_tehran_pop[
            self.df_tehran_pop['population'] != 0
        ].reset_index()
        # self.df_tehran_pop['population'] = self.df_tehran_pop['population'].map( lambda x: int(float(str(x).replace(',',''))) )
        self.df_tehran_pop['tile_id'] = self.df_tehran_pop.index
        self.df_tehran_pop = self.df_tehran_pop[['tile_id', 'population', 'lat', 'lng']]
        self.df_tehran_pop_gpd = gpd.GeoDataFrame(
            self.df_tehran_pop,
            geometry = gpd.points_from_xy(
                self.df_tehran_pop.lng,
                self.df_tehran_pop.lat
            )
        )
        ########################################################################
        # Return
        return
    def generate( self, start_time, end_time, n_users, verbose = False ):
        # Report
        if( verbose ):
            n_records_generated = 0
            print(
                'Progress: {:>3}/{:<3}, Records Generated: {:>12}'.format(
                    0,
                    n_users,
                    n_records_generated
                ),
                end = '\r',
                flush = True
            )
        # Main
        dfs = []
        starting_locations = np.random.choice(
            self.df_tehran_pop_gpd['tile_id'],
            size = n_users,
            replace = True,
            p = self.df_tehran_pop_gpd['population'] / self.df_tehran_pop_gpd['population'].sum()
        )
        for i, starting_location in zip( range( n_users ), starting_locations ):
            id_user = i + 1
            # Generate
            df = self.ditras.generate(
                start_time,
                end_time,
                self.df_tehran_pop_gpd,
                starting_locations = [ starting_location ],
                relevance_column = 'population',
                show_progress = False
            )
            # Set ID User
            df['uid'] = id_user
            # Append
            dfs.append( df )
            # Report
            if( verbose ):
                n_records_generated += len(df)
                print(
                    'Progress: {:>3}/{:<3}, Records Generated: {:>12}'.format(
                        i + 1,
                        n_users,
                        n_records_generated
                    ),
                    end = '\r'
                )
        # Report
        if( verbose ):
            print(
                'Progress: {:>3}/{:<3}, Records Generated: {:>12}'.format(
                    i + 1,
                    n_users,
                    n_records_generated
                ),
                flush = True
            )
        # Concatenate
        df_trajectory_raw_latlng = pd.concat( dfs )
        # Rename
        df_trajectory_raw_latlng.rename(
            columns = {'uid': 'id_user'},
            inplace = True
        )
        # Create UnixTimeStamp
        df_trajectory_raw_latlng['timestamp'] = pd.to_datetime( df_trajectory_raw_latlng['datetime'] ).astype( np.int64 ) // 10**9
        # Re-Select
        df_trajectory_raw_latlng = df_trajectory_raw_latlng[['id_user', 'timestamp', 'lat', 'lng']]
        # Sort & Re-Index
        df_trajectory_raw_latlng.sort_values(['id_user', 'timestamp'], inplace = True)
        df_trajectory_raw_latlng.reset_index( drop = True, inplace = True )
        # Return
        return( df_trajectory_raw_latlng )
































