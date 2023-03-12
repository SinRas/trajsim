"""Trajectory Generation Module
"""

# Modules
import datetime
import random
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from skmob.utils import constants

import skmob
from skmob.models.epr import Ditras
from skmob.models.markov_diary_generator import MarkovDiaryGenerator
from skmob.preprocessing import filtering, compression, detection, clustering

# DEBUG
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Parameters

# Methods

# Classes
## MarkovDiaryGenerator from Matrix
class MDGMatrix(MarkovDiaryGenerator):
    """Markov Diary Generator constructed using a matrix of 48*48.
    """
    # Constructor
    def __init__(self, fp_matrix, name='Matrix Markov Diary'):
        super().__init__(name)
        self.time_resolution = 60*60
        self.fp_matrix = fp_matrix
        self.smdg = pd.read_csv(self.fp_matrix, header = None).values  # TODO: specific formatting
        self._create_empty_markov_chain()
        n_rows, n_cols = self.smdg.shape
        for i in range(n_rows):
            for j in range(n_cols):
                # Skip Nans
                if( np.isnan( self.smdg[i,j] ) or self.smdg[i,j] == 0 ):
                    continue
                # Extract Info
                ## Home/Familiar: 1, Non-Familiar: 0
                ## Location
                loc = 1 if i < 24 else 0
                loc_next = 1 if j < 24 else 0
                ## Hour
                hour = i % 24
                hour_next = j % 24
                # Update
                self.markov_chain_[(hour,loc)][(hour_next,loc_next)] = self.smdg[i,j]
        self._normalize_markov_chain()
        return
## MarkovDiaryGenerator from Ditras
class MDGDitras(MarkovDiaryGenerator):
    """Markov Diary Generator constructed using data provided by DITRAS paper.
    """
    # Constructor
    def __init__(self, fp_ditras_diary, name='DITRAS Markov Diary'):
        super().__init__(name)
        self.time_resolution = 60*60
        self.fp_ditras_diary = fp_ditras_diary
        with open(self.fp_ditras_diary, 'rb') as in_file:
            self.diary_generator = pickle.load(in_file)
        self._create_empty_markov_chain()
        for (hour, loc), entry in self.diary_generator.items():
            for (hour_next, loc_next), prob in entry.items():
                self.markov_chain_[(hour,loc)][(hour_next,loc_next)] = prob
        self._normalize_markov_chain()
        return
## MarkovDiaryGenerator for 30min
class MDGDitras30min(MDGDitras):
    # Constructor
    def __init__(self, fp_ditras_diary, name='DITRAS Markov Diary'):
        self._time_slot_length = '30min'
        self.time_resolution = 30*60
        self.T = 24*2
        self.dt = datetime.timedelta(minutes=30)
        super().__init__(fp_ditras_diary, name)
        return
    # Create Empty Markov Chain
    def _create_empty_markov_chain(self):
        """
        Create an empty Markov chain, i.e., a matrix (2*48) * (2*48) where an element M(i,j) is a pair of pairs
        ((h_i, b_i), (h_j, b_j)), h_i, h_j \in {0, ..., 23, ..., 47} and b_i, b_j \in {0, 1}
        """
        self._markov_chain_ = defaultdict(lambda: defaultdict(float))
        for h1 in range(0, self.T):
            for r1 in [0, 1]:
                for h2 in range(0, self.T):
                    for r2 in [0, 1]:
                        self._markov_chain_[(h1, r1)][(h2, r2)] = 0.0
    # Update Markov Chain
    def _update_markov_chain(self, time_series, shift=0):
        """
        Update the Markov Chain by including the behavior of an individual
        
        Parameters
        ----------
        time_series: pandas DataFrame
            time series of abstract locations visisted by an individual.
        """
        HOME = 1
        TYPICAL, NON_TYPICAL = 1, 0

        n = len(time_series)  # n is the length of the time series of the individual
        slot = 0  # it starts from the first slot in the time series

        while slot < n - 1:  # scan the time series of the individual, time slot by time slot

            #h = (slot % 24)
            h = (slot  + shift) % self.T  # h, the hour of the day
            next_h = (h + 1) % self.T  # next_h, the next hour of the day

            loc_h = time_series[slot]  # loc_h  ,   abstract location at the current slot
            next_loc_h = time_series[slot + 1]  # d_{h+1},   abstract location at the next slot

            if loc_h == HOME:  # if \delta(loc_h, t_h) == 1, i.e., she stays at home

                # we have two cases
                if next_loc_h == HOME:  # if \delta(d_{h + 1}, t_{h + 1}) == 1

                    # we are in Type1: (h, 1) --> (h + 1, 1)
                    self._markov_chain_[(h, TYPICAL)][(next_h, TYPICAL)] += 1

                else:  # she will be not in the typical location

                    # we are in Type2: (h, 1) --> (h + tau, 0)
                    tau = 1
                    if slot + 2 < n:  # if slot is the second last in the time series

                        for j in range(slot + 2, n):  # in slot + 1 we do not have HOME so we start from slot + 2
                            loc_hh = time_series[j]
                            if loc_hh == next_loc_h:  # if \delta(d_{h + j}, d_{h + 1}) == 1
                                tau += 1
                            else:
                                break

                        h_tau = (h + tau) % self.T
                        # update the state of edge (h, 1) --> (h + tau, 0)
                        self._markov_chain_[(h, TYPICAL)][(h_tau, NON_TYPICAL)] += 1
                        slot = j - 2 #1

                    else:  # terminate the while cycle
                        slot = n

            else:  # loc_h != HOME

                if next_loc_h == HOME:  # if \delta(d_{h + 1}, t_{h + 1}) == 1, i.e., she will stay at home

                    # we are in Type3: (h, 0) --> (h + 1, 1)
                    self._markov_chain_[(h, NON_TYPICAL)][(next_h, TYPICAL)] += 1

                else:

                    # we are in Type 4: (h, 0) --> (h + tau, 0)
                    tau = 1
                    if slot + 2 < n:

                        for j in range(slot + 2, n):
                            loc_hh = time_series[j]
                            if loc_hh == next_loc_h:  # if \delta(d_{h + j}, d_{h + 1}) == 1
                                tau += 1
                            else:
                                break

                        h_tau = (h + tau) % self.T

                        # update the state of edge (h, 0) --> (h + tau, 0)
                        self._markov_chain_[(h, NON_TYPICAL)][(h_tau, NON_TYPICAL)] += 1
                        slot = j - 2 #1

                    else:
                        slot = n

            slot += 1
        return
    # Generate Trajectory
    def generate(self, diary_length, start_date, random_state=None):
        """
        Start the generation of the mobility diary.
        
        Parameters
        ----------
        diary_length : int
            the length of the diary in hours.

        start_date : datetime
            the starting date of the generation.
        
        Returns
        -------
        pandas DataFrame
            the generated mobility diary.
        """
        current_date = start_date
        V, i = [], 0
        prev_state = (i, 1)  # it starts from the typical location at midnight
        V.append(prev_state)
        
        if random_state is not None:
            random.seed(random_state)

        while i < diary_length:

            h = i % self.T  # the hour of the day

            # select the next state in the Markov chain
            p = list(self._markov_chain_[prev_state].values())
            if sum(p) == 0.:
                hh, rr = prev_state
                next_state = ((hh + 1) % self.T, rr)
            else:
                index = self._weighted_random_selection(p)
                next_state = list(self._markov_chain_[prev_state].keys())[index]
            V.append(next_state)

            j = next_state[0]
            if j > h:  # we are in the same day
                i += j - h
            else:  # we are in the next day
                i += self.T - h + j

            prev_state = next_state

        # now we translate the temporal diary into the the mobility diary
        prev, diary, other_count = V[0], [], 1
        diary.append([current_date, 0])

        for v in V[1:]:  # scan all the states obtained and create the synthetic time series
            h, s = v
            h_prev, s_prev = prev

            if s == 1:  # if in that hour she visits home
                current_date += self.dt
                diary.append([current_date, 0])
                other_count = 1
            else:  # if in that hour she does NOT visit home

                if h > h_prev:  # we are in the same day
                    j = h - h_prev
                else:  # we are in the next day
                    j = self.T - h_prev + h

                for i in range(0, j):
                    current_date += self.dt
                    diary.append([current_date, other_count])
                other_count += 1

            prev = v

        short_diary = []
        prev_location = -1
        for visit_date, abstract_location in diary[0: diary_length]:
            if abstract_location != prev_location:
                short_diary.append([visit_date, abstract_location])
            prev_location = abstract_location

        diary_df = pd.DataFrame(short_diary, columns=[constants.DATETIME, 'abstract_location'])
        return diary_df
    #################################################################################
    #################################################################################
    #################################################################################
    #################################################################################
    # Generate Trajectory
    def generate_by_enddate(self, start_date, end_date, random_state=None):
        """
        Start the generation of the mobility diary.
        
        Parameters
        ----------
        start_date : datetime
            the starting date of the generation.
        
        end_date : datetime
            the ending date of the generation.
        
        Returns
        -------
        pandas DataFrame
            the generated mobility diary.
        """
        current_date = start_date
        V, i = [], 0
        prev_state = (i, 1)  # it starts from the typical location at midnight
        V.append(prev_state)
        
        if random_state is not None:
            random.seed(random_state)

        while current_date < end_date:

            h = i % self.T  # the hour of the day

            # select the next state in the Markov chain
            p = list(self._markov_chain_[prev_state].values())
            if sum(p) == 0.:
                hh, rr = prev_state
                next_state = ((hh + 1) % self.T, rr)
            else:
                index = self._weighted_random_selection(p)
                next_state = list(self._markov_chain_[prev_state].keys())[index]
            V.append(next_state)

            j = next_state[0]
            if j >= h:  # we are in the same day
                i += j - h
                current_date += (j-h)*self.dt
            else:  # we are in the next day
                i += self.T - h + j
                current_date += (self.T - h + j)*self.dt
            prev_state = next_state
        # now we translate the temporal diary into the the mobility diary
        current_date = start_date
        prev, diary, other_count = V[0], [], 1
        diary.append([current_date, 0])
        for v in V[1:]:  # scan all the states obtained and create the synthetic time series
            h, s = v
            h_prev, s_prev = prev

            if s == 1:  # if in that hour she visits home
                current_date += self.dt
                diary.append([current_date, 0])
                other_count = 1
            else:  # if in that hour she does NOT visit home

                if h > h_prev:  # we are in the same day
                    j = h - h_prev
                else:  # we are in the next day
                    j = self.T - h_prev + h

                for i in range(0, j):
                    current_date += self.dt
                    diary.append([current_date, other_count])
                other_count += 1

            prev = v

        short_diary = []
        prev_location = -1
        for visit_date, abstract_location in diary:
            if visit_date > end_date:
                break
            if abstract_location != prev_location:
                short_diary.append([visit_date, abstract_location])
            prev_location = abstract_location
        diary_df = pd.DataFrame(short_diary, columns=[constants.DATETIME, 'abstract_location'])
        return diary_df
## GeoPandasPoints
class GeoPandasTehran:
    # Constructor
    def __init__(self, fp_points_tehran, rescale_relevance = False):
        self.rescale_relevance = rescale_relevance
        self.fp_points_tehran = fp_points_tehran
        self.df = pd.read_csv(self.fp_points_tehran)
        cols = set(self.df.columns)
        if not 'relevance' in cols and 'weight' in cols:
            self.df['relevance'] = self.df['weight']
        if not 'tile_id' in cols:
            self.df['tile_id'] = self.df.index
        if not 'location_name' in cols:
            self.df['location_name'] = self.df.index.astype(str)
        self.df = self.df[['tile_id', 'relevance', 'lat', 'lng', 'location_name']]
        # Re-Scale
        if self.rescale_relevance:
            self.df['relevance'] = self.df['relevance'] / (100*self.df['relevance'].min())
        self.df_gpd = gpd.GeoDataFrame(
            self.df,
            geometry = gpd.points_from_xy(
                self.df.lng,
                self.df.lat
            )
        )
        return

## Base
class DitrasTimeFlexible(Ditras):
    # Generate Agent, Time Flexible
    def _epr_generate_one_agent(self, agent_id, start_date, end_date):

        # generate a mobility diary for the agent
        rand_seed_diary = np.random.randint(0,10**6)
        diary_df = self._diary_generator.generate_by_enddate(start_date, end_date, random_state=rand_seed_diary)

        for i, row in diary_df.iterrows():
            if row.abstract_location == 0:  # the agent is at home
                self._trajectories_.append((agent_id, row.datetime, self._starting_loc))
                self._location2visits[self._starting_loc] += 1

            else:  # the agent is not at home
                next_location = self._choose_location()
                self._trajectories_.append((agent_id, row.datetime, next_location))
                self._location2visits[next_location] += 1

class TrajectoryGenerationDITRAS:
    """Trajectory Generation using DITRAS.
    """
    # Constructor
    def __init__( self, mdg_or_fp_ditras_diary, gpd_or_fp_tehran_points ):
        ########################################################################
        # Markov Diary Generator
        if isinstance(mdg_or_fp_ditras_diary, MarkovDiaryGenerator):
            self.mdg = mdg_or_fp_ditras_diary
        elif isinstance(mdg_or_fp_ditras_diary, str):
            self.mdg = MDGDitras(fp_ditras_diary=mdg_or_fp_ditras_diary)
        else:
            raise ValueError("`mdg_or_fp_ditras_diary` should be either of type 'skmob.models.markov_diary_generator.MarkovDiaryGenerator' or 'path to ditras diary'")
        self.ditras = DitrasTimeFlexible( self.mdg, rho = 0.6, gamma = 0.21 )  # Parameters based on paper: https://link.springer.com/article/10.1007/s10618-017-0548-4
        ########################################################################
        # Tessellation Load
        if isinstance(gpd_or_fp_tehran_points, gpd.GeoDataFrame):
            self.df_gpd = gpd_or_fp_tehran_points
        elif isinstance(gpd_or_fp_tehran_points, str):
            self.tehran_gpd = GeoPandasTehran(fp_points_tehran=gpd_or_fp_tehran_points)
            self.df_gpd = self.tehran_gpd.df_gpd
        else:
            raise ValueError("`gpd_or_fp_tehran_points` should be either of type 'gpd.GeoDataFrame' or 'path to tehran points file'")
        # Return
        return
    # Generate Trajectories
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
            self.df_gpd['tile_id'],
            size = n_users,
            replace = True,
            p = self.df_gpd['relevance'] / self.df_gpd['relevance'].sum()
        )
        for i, starting_location in zip( range( n_users ), starting_locations ):
            id_user = i + 1
            # Generate
            df = self.ditras.generate(
                start_time,
                end_time,
                self.df_gpd,
                relevance_column = 'relevance',
                show_progress = False,
                n_agents = 1,
                starting_locations=[starting_location]
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
        df_trajectory_raw_latlng = df_trajectory_raw_latlng[['id_user', 'timestamp', 'lat', 'lng', 'datetime']]
        # Sort & Re-Index
        df_trajectory_raw_latlng.sort_values(['id_user', 'timestamp'], inplace = True)
        df_trajectory_raw_latlng.reset_index( drop = True, inplace = True )
        # Return
        return( df_trajectory_raw_latlng )
































