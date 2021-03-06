"""Trajectory Hashing methods.

Methods to create hashes for each trajectory for fast comparison of trajectories.
"""
# Modules
import sys, random
from typing import Set, List
import numpy as np
import pandas as pd

import pyspark.sql.functions as sql_functions
import pyspark.sql.types as sql_types
from pyspark.sql import Window
# Parameters
# HASHPRIME = 11
# HASHPRIME = 101
# HASHPRIME = 5011
HASHPRIME = 2038074743
# HASHPRIME = 988666444411111

# Methods

# Classes
## Base
class TrajectoryHasherBase:
    """Base trajectory hasher class.
    
    Methods to implement:
    - _initialize
    - hash
    """
    # Constructor
    def __init__( self, name, **params ):
        # Parameters
        self.name = name
        self.params = params
        # Initialize
        self._initialize()
        # Return
        return
    # Initialize
    def _initialize( self ):
        Exception( '<initialize@TrajectoryHasherBase> : not implemented!' )
        return
    # Bucketize
    def hash( self, df_trajectory_processed ):
        Exception( '<hash@TrajectoryHasherBase> : not implemented!' )
        return # df_user_hashes
## Jacard Estimation SinRas
class TrajectoryHasherJacardEstimation(TrajectoryHasherBase):
    """Trajectory Hashing using Jacard Similarity Estimation.
    """
    # Constructor
    def __init__(
            self,
            name,
            n_hashes,
            **params
        ):
        # Parameters
        self.n_hashes = n_hashes
        super().__init__( name, **params )
        # Return
        return
    # Initialize
    def _initialize( self ):
        return
    # Hash
    def hash( self, df_trajectory_processed, df_type = 'pandas' ):
        # Assert Implemented Methods
        assert df_type in { 'pandas', 'spark' }, 'hash@<TrajectoryHasherJacardEstimation>: df_type = "{}" is not implemented!'.format( df_type )
        # Hash
        if( df_type == 'pandas' ):
            # Bounds
            id_timestamp_min = df_trajectory_processed['id_timestamp'].min()
            id_timestamp_max = df_trajectory_processed['id_timestamp'].max()
            # Select ID TimeStamp for Hashes
            id_timestamps_selected = np.random.choice(
                np.arange( id_timestamp_min, id_timestamp_max+1 ),
                self.n_hashes,
                replace = False
            ) if self.n_hashes < (id_timestamp_max - id_timestamp_min + 1) else list(range( id_timestamp_min, id_timestamp_max+1 ))
            id_timestamps_selected_set = set(id_timestamps_selected)
            # Filter
            df_result = df_trajectory_processed[
                df_trajectory_processed['id_timestamp'].map(
                    lambda x:  x in id_timestamps_selected_set
                )
            ].copy().sort_values(['id_user', 'id_timestamp'])
            # Index Locations
            location_indices = []
            lat_lng_to_idx = dict()
            for lat, lng in zip( df_result['lat'], df_result['lng'] ):
                key = (lat,lng)
                if( not key in lat_lng_to_idx ):
                    lat_lng_to_idx[key] = len(lat_lng_to_idx)
                # Add New Index
                location_indices.append( lat_lng_to_idx[key] )
            df_result['location_indices'] = location_indices
            # Calculate Hashes
            df_hashes = None
            for i, id_timestamp in enumerate(id_timestamps_selected):
                if( df_hashes is None ):
                    df_hashes = df_result[
                        df_result['id_timestamp'] == id_timestamp
                    ][['id_user', 'location_indices']].copy()
                else:
                    df_hashes['location_indices'] = df_result[
                        df_result['id_timestamp'] == id_timestamp
                    ]['location_indices'].values
                # Rename New Column
                colname = 'hash_{}'.format( i )
                df_hashes.rename(columns={'location_indices':colname}, inplace = True)
        elif( df_type == 'spark' ):
            # ID TimeStamp Bounds
            row = df_trajectory_processed.agg(
                sql_functions.min( sql_functions.col("id_timestamp") ).alias("id_timestamp_min"),
                sql_functions.max( sql_functions.col("id_timestamp") ).alias("id_timestamp_max")
            ).head()
            id_timestamp_min, id_timestamp_max = row['id_timestamp_min'], row['id_timestamp_max']
            # Chosen TimeStamps
            id_timestamps_selected = np.random.choice(
                np.arange( id_timestamp_min, id_timestamp_max+1 ),
                self.n_hashes,
                replace = False
            ).tolist() if self.n_hashes < (id_timestamp_max - id_timestamp_min + 1) else list(range( id_timestamp_min, id_timestamp_max+1 ))
            # Create SparkDataFrame
            df_id_timestamps = self.params['spark'].createDataFrame(
                [ [id_timestamp, 'hash_{}'.format(i)] for i, id_timestamp in enumerate(id_timestamps_selected) ],
                schema = sql_types.StructType([
                    sql_types.StructField( 'id_timestamp', sql_types.IntegerType(), False ),
                    sql_types.StructField( 'hash_name', sql_types.StringType(), False ),
                ])
            )
            # ID Locations
            df_location_ids = df_trajectory_processed.select("lat", "lng").\
                distinct().withColumn(
                    "id_location",
                    sql_functions.row_number().over(
                        Window.orderBy("lat", "lng")
                    )
                )
            # Join Hashes
            df_result = df_trajectory_processed.join(
                df_id_timestamps,
                on = [ 'id_timestamp' ],
                how = 'inner'
            ).join(
                df_location_ids,
                on = [ 'lat', 'lng' ],
                how = 'inner'
            )
            # Turn Into Table
            df_hashes = df_result.groupby("id_user").\
                pivot( "hash_name" ).\
                agg( sql_functions.first( "id_location" ) )
        # Return
        return( df_hashes )
    # Estimate Similarity
    def estimate_similarity( self, df_trajectory_processed, df_type = 'pandas' ):
        # Assert Implemented Methods
        assert df_type in { 'pandas' }, 'estimate_similarity@<TrajectoryHasherJacardEstimation>: df_type = "{}" is not implemented!'.format( df_type )
        # Estimate
        if( df_type == 'pandas' ):
            # Get Hashes
            df_hashes = self.hash( df_trajectory_processed, df_type = 'pandas' )
            # Extract
            ## ID Users
            id_users = df_hashes['id_user'].tolist()
            ## Hashes
            hashes_np = df_hashes[
                [ x for x in df_hashes.columns if x.startswith('hash_') ]
            ].values
            # Loop Over Users
            result = []
            for i, (id_user, hash_vector) in enumerate(zip( id_users, hashes_np )):
                # Skip Last Row
                if( i == (len(id_users) - 1) ):
                    continue
                #
                for j, n_common in enumerate( np.sum( hashes_np[(i+1):] == hash_vector, axis = 1 ) ):
                    id_user_other = id_users[ (j+i+1) ]
                    result.append( [ id_user, id_user_other, n_common / (2*self.n_hashes) ] ) # DEBUG! WHY 2* DENOMINATOR?
            # Convert to DataFrame
            df_user_estimated_similarities = pd.DataFrame(
                result,
                columns = [ 'id_user', 'id_user_other', 'similarity_estimated' ]
            )
        # Return
        return( df_user_estimated_similarities )
        
## MinHash Mohsen Implementation
### Auxilary Class: Calculator
class Calculator:
    # Reverse Bytes
    @staticmethod
    def reverse_bytes( _int: int, _n_bytes: int = 8 ) -> int:
        result = int.from_bytes(
            _int.to_bytes( _n_bytes, sys.byteorder )[::-1],
            sys.byteorder
        )
        # Return
        return( result )
    # Constructor
    def __init__( self, a: int, b: int ):
        self.a = a
        self.b = b
    # Hash
    def hash( self, _input: List[int] ) -> int:
        # Validity Check
        assert len(_input) > 0, 'Input size should not be empty.'
        # MinHash
        minhash = ( self.a * _input[0] + self.b ) % HASHPRIME
        for x in _input[1:]:
            h = ( self.a * x + self.b ) % HASHPRIME
            minhash = min( minhash, h )
        # Result
        # result = Calculator.reverse_bytes( minhash, 6 )
        result = minhash
        # Return
        return( result )
## Main
class TrajectoryHasherMinHash(TrajectoryHasherBase):
    """Trajectory Hashing using MinHash Similarity.
    """
    # Statics
    ## Class Parameters
    
    ## Methods
    ### Similarity
    @staticmethod
    def similarity( set_a: Set[int], set_b: Set[int] ) -> float:
        n_intersection = len( set_a.intersection( set_b ) )
        n_union = len(set_a) + len(set_b) - n_intersection
        return( n_intersection / n_union )

    ### Random Calculator
    @staticmethod
    def random_calculators( n_calculators: int, seed: int = 0 ):
        # Set Seed
        random.seed( seed )
        # Result
        result = [
            Calculator(
                1 + random.randrange( HASHPRIME - 1 ),
                random.randrange( HASHPRIME )
            ) for _ in range( n_calculators )
        ]
        # Return
        return( result )
    ### Binary Representation Padding
    @staticmethod
    def bin_padded( _int, n = 32 ):
        assert isinstance(_int, int) and _int >= 0, 'Should input a non-negative integer!'
        res = bin(_int)[2:]
        res = (n - len(res)) * '0' + res
        return( res )
    ### Convert TimeStamp(index) and Lats/Lngs to Integer Representation
    @staticmethod
    def t_and_cell_int( ts, lats, lngs, lat_lng_to_location_id ):
        #
        result = np.zeros( len(ts), dtype = np.int64 )
        for i, (t, lat, lng) in enumerate( zip( ts, lats, lngs ) ):
            cell = lat_lng_to_location_id[(lat, lng)]
            result[i] = int(
                TrajectoryHasherMinHash.bin_padded( t )[-16:] + \
                TrajectoryHasherMinHash.bin_padded(cell, 16)[-16:],
                2
            )
        return( result )
    
    # Constructor
    def __init__(
            self,
            name,
            n_hashes,
            seed,
            **params
        ):
        # Parameters
        self.n_hashes = n_hashes
        self.seed = seed
        super().__init__( name, **params )
        # Return
        return
    # Initialize
    def _initialize( self ):
        self.calculators = TrajectoryHasherMinHash.random_calculators(
            n_calculators = self.n_hashes,
            seed = self.seed
        )
        return
    # Bucketize
    def hash( self, df_trajectory_processed, df_type = 'pandas', return_features_list = False ):
        # Assert Implemented Methods
        assert df_type in { 'pandas', 'spark' }, 'hash@<TrajectoryHasherMinHash>: df_type = "{}" is not implemented!'.format( df_type )
        # Hash
        if( df_type == 'pandas' ):
            # Create Lat/Lng to ID Dictionary
            lat_lng_uniques = np.unique(
                df_trajectory_processed[['lat','lng']].values,
                axis = 1
            )
            lat_lng_to_location_id = {
                tuple(x): i for i,x in enumerate( lat_lng_uniques )
            }
            del lat_lng_uniques
            # Add Features as Ints
            df_trajectory_processed['t_and_cell_int'] = \
                TrajectoryHasherMinHash.t_and_cell_int(
                    df_trajectory_processed['id_timestamp'],
                    df_trajectory_processed['lat'],
                    df_trajectory_processed['lng'],
                    lat_lng_to_location_id
            )
            # Add Hashes
            id_users = df_trajectory_processed['id_user'].unique()
            results_hashes = np.zeros( (len(id_users), self.n_hashes), dtype = np.int64 )
            results_id_users = np.zeros( len(id_users), dtype = np.int64 )
            if( return_features_list ):
                results_t_and_cell_int_list = []
            # Main Loop
            for i, id_user in enumerate(id_users):
                # Store Id User
                results_id_users[i] = id_user
                # Get User Features
                features_int_id_user_list = df_trajectory_processed[
                    df_trajectory_processed['id_user'] == id_user
                ]['t_and_cell_int'].tolist()
                # Add Features List
                if( return_features_list ):
                    results_t_and_cell_int_list.append( features_int_id_user_list )
                # Calculate Hashes
                for j, calculator in enumerate(self.calculators):
                    hash_int = calculator.hash( features_int_id_user_list )
                    results_hashes[i][j] = hash_int
            # Convert to DataFrame
            _dict = {
                'id_user': id_users
            }
            _dict.update({
                'hash_{}'.format(i): v for i, v in enumerate( results_hashes.T )
            })
            ## Add Features List
            if( return_features_list ):
                _dict.update({
                    't_and_cell_int_list': results_t_and_cell_int_list
                })
            df_hashes = pd.DataFrame( _dict )
        elif( df_type == 'spark' ):
            # ID Locations
            df_location_ids = df_trajectory_processed.select("lat", "lng").\
                distinct().withColumn(
                    "id_location",
                    sql_functions.row_number().over(
                        Window.orderBy("lat", "lng")
                    )
                )
            # Ratio
            n_id_locations = df_location_ids.count()
            alpha = 10 ** ( int(np.log10( n_id_locations )) + 1 )
            # Add Integer Feature
            df_int_featured = df_trajectory_processed.join(
                df_location_ids,
                on = [ 'lat', 'lng' ],
                how = 'inner'
            ).select(
                "id_user",
                (
                    sql_functions.col("id_timestamp") * alpha + \
                    sql_functions.col("id_location")
                ).cast("long").alias("int_feature")
            ).repartition( 64 )
            # As and Bs
            As = [ 1 + random.randrange( HASHPRIME - 1 ) for _ in range(self.n_hashes) ]
            Bs = [ random.randrange( HASHPRIME ) for _ in range(self.n_hashes) ]
            # Main
            aggs = [
                sql_functions.min( sql_functions.col("hash_{}".format(i)) ).alias("hash_{}".format(i)) for i in range(self.n_hashes)
            ]
            df_hashes = df_int_featured.select(*(['id_user'] + [
                ( (a*sql_functions.col("int_feature")+b)%HASHPRIME ).cast("int").alias('hash_{}'.format(i)) for i,(a,b) in enumerate(zip(As,Bs))
            ])).groupby("id_user").agg( *aggs )
        # Return
        return( df_hashes )
    # Estimate Similarity
    def estimate_similarity( self, df_trajectory_processed, df_type = 'pandas' ):
        # Assert Implemented Methods
        assert df_type in { 'pandas' }, 'estimate_similarity@<TrajectoryHasherMinHash>: df_type = "{}" is not implemented!'.format( df_type )
        # Estimate
        if( df_type == 'pandas' ):
            # Get Hashes
            df_hashes = self.hash( df_trajectory_processed, df_type = 'pandas' )
            # Extract
            ## ID Users
            id_users = df_hashes['id_user'].tolist()
            ## Hashes
            hashes_np = df_hashes[
                [ x for x in df_hashes.columns if x.startswith('hash_') ]
            ].values
            # Loop Over Users
            result = []
            for i, (id_user, hash_vector) in enumerate(zip( id_users, hashes_np )):
                # Skip Last Row
                if( i == (len(id_users) - 1) ):
                    continue
                #
                for j, n_common in enumerate( np.sum( hashes_np[(i+1):] == hash_vector, axis = 1 ) ):
                    id_user_other = id_users[ (j+i+1) ]
                    result.append( [ id_user, id_user_other, n_common / self.n_hashes ] ) # DEBUG! WHY 2* DENOMINATOR?
            # Convert to DataFrame
            df_user_estimated_similarities = pd.DataFrame(
                result,
                columns = [ 'id_user', 'id_user_other', 'similarity_estimated' ]
            )
        # Return
        return( df_user_estimated_similarities )