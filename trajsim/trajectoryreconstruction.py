"""Trajectory Reconstruction methods.

Methods that get a non-regular-stepped trajectory and output a fixed-step trajectory.
"""
# Modules
import numpy as np
import pandas as pd

import pyspark.sql.functions as sql_functions
import pyspark.sql.types as sql_types
from pyspark.sql import Window
# Parameters
SCHEMA_DATA_POINT = sql_types.StructType([
    sql_types.StructField( 'id_timestamp', sql_types.IntegerType(), False ),
    sql_types.StructField( 'lat', sql_types.DoubleType(), False ),
    sql_types.StructField( 'lng', sql_types.DoubleType(), False ),
])

# Methods
## Gather Data Points
def gather_data_point( id_timestamp, lat, lng ):
    return( (id_timestamp, lat, lng) )
udf_gather_data_point = sql_functions.udf(
    gather_data_point,
    SCHEMA_DATA_POINT
)

# Classes
## Base
class TrajectoryReconstructorBase:
    """Base trajectory reconstruction class.
    
    Methods to implement:
    - _initialize
    - reconstruct
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
        Exception( '<initialize@TrajectoryReconstructorBase> : not implemented!' )
        return
    # Reconstruct
    def reconstruct( self, df_trajectory_bucketed ):
        Exception( '<reconstruct@TrajectoryReconstructorBase> : not implemented!' )
        return # df_trajectory_bucketed_reconstructed
## Linear Reconstructor
class TrajectoryReconstructorLinear( TrajectoryReconstructorBase ):
    """Reconstruct or Regularize Trajectory Linearly (e.g. Linear Imputation).
    
    The gap between timestamp ids is filled by linear interpolation of lat/lng
    before and after that.
    """
    # Constructor
    def __init__( self, name, **params ):
        super().__init__( name, **params )
        return
    # Initialize
    def _initialize( self ):
        return
    # Reconstruct
    def reconstruct( self, df_trajectory_bucketed, df_type = 'pandas' ):
        # Assert Implemented Methods
        assert df_type in { 'pandas', 'spark' }, 'reconstruct@<TrajectoryReconstructorLinear>: df_type = "{}" is not implemented!'.format( df_type )
        # Reconstruct
        if( df_type == 'pandas' ):
            # Sort and Copy
            df_result = df_trajectory_bucketed.copy().sort_values(['id_user', 'id_timestamp'])
            df_result.reset_index( drop = True, inplace = True )
            # ID TimeStamp Bounds
            id_timestamp_min = df_result['id_timestamp'].min()
            id_timestamp_max = df_result['id_timestamp'].max()
            # New Data
            data = []
            _n = len(df_result)
            for i, (id_user, id_timestamp, lat, lng) in enumerate(zip(
                    df_result['id_user'], df_result['id_timestamp'],
                    df_result['lat'], df_result['lng']
                )):
                # First Row and Missing TimeStamps
                if( i == 0 and id_timestamp > id_timestamp_min ):
                    data.extend([
                        [id_user, t, lat, lng] for t in range(
                            id_timestamp_min, id_timestamp
                        )
                    ])
                # Get Next Row if Possible
                if( i < (_n-1) ):
                    # Next
                    id_user_next = df_result['id_user'][i+1]
                    id_timestamp_next = df_result['id_timestamp'][i+1]
                    lat_next = df_result['lat'][i+1]
                    lng_next = df_result['lng'][i+1]
                    # Linear Interpolation
                    if( id_user == id_user_next and (id_timestamp+1) < id_timestamp_next ):
                        # Line Slopes
                        slope_lat = ( lat_next - lat ) / ( id_timestamp_next - id_timestamp )
                        slope_lng = ( lng_next - lng ) / ( id_timestamp_next - id_timestamp )
                        # Update
                        for t in range( id_timestamp+1, id_timestamp_next ):
                            dt = ( t - id_timestamp )
                            data.append([
                                id_user, t, lat + dt * slope_lat, lng + dt * slope_lng
                            ])
                    elif( id_user != id_user_next ):
                        # New User Encountered
                        data.extend([
                            [id_user, t, lat, lng] for t in range(
                                id_timestamp, id_timestamp_max + 1
                            )
                        ])
                        data.extend([
                            [id_user_next, t, lat, lng] for t in range(
                                id_timestamp_min, id_timestamp_next
                            )
                        ])
                else:
                    # Last Row and Missing Following TimeStamps
                    if( id_timestamp < id_timestamp_max ):
                        data.extend([
                            [id_user, t, lat, lng] for t in range(
                                (id_timestamp+1), (id_timestamp_max+1)
                            )
                        ])
            # Append New Data
            df_result = pd.concat([
                df_result,
                pd.DataFrame(
                    data,
                    columns = [ 'id_user', 'id_timestamp', 'lat', 'lng' ]
                )
            ]).sort_values(['id_user', 'id_timestamp']).reset_index( drop = True )
        elif( df_type == 'spark' ):
            # ID TimeStamp Bounds
            row = df_trajectory_bucketed.agg(
                sql_functions.min( sql_functions.col("id_timestamp") ).alias("id_timestamp_min"),
                sql_functions.max( sql_functions.col("id_timestamp") ).alias("id_timestamp_max")
            ).head()
            id_timestamp_min, id_timestamp_max = row['id_timestamp_min'], row['id_timestamp_max']
            # Define Imputer Function
            def linear_interpolator( x_arr ):
                # Result
                result = []
                # Sort by TimeStamp
                x_arr = sorted( x_arr, key = lambda x: x[0] )
                
                # First TimeStamp
                id_timestamp_first, lat_first, lng_first = x_arr[0]
                for id_timestamp in range( id_timestamp_min, id_timestamp_first ):
                    result.append( (id_timestamp, lat_first, lng_first) )
                
                # Linear Reconstruction
                for data_now, data_next in zip( x_arr[:-1], x_arr[1:] ):
                    # Extract Info
                    id_timestamp_now, lat_now, lng_now = data_now
                    id_timestamp_next, lat_next, lng_next = data_next
                    # Add Now
                    result.append( (id_timestamp_now, lat_now, lng_now) )
                    # Skip Linear Interpolation
                    assert id_timestamp_next > id_timestamp_now, 'Sorting has gone wrong!'
                    if( (id_timestamp_next - id_timestamp_now) == 1 ):
                        continue
                    # Linear Interpolation
                    ## Slopes
                    dt = id_timestamp_next - id_timestamp_now
                    slope_lat = (lat_next - lat_now) / dt
                    slope_lng = (lng_next - lng_now) / dt
                    ## Add Inner Points
                    for i in range(  1, id_timestamp_next - id_timestamp_now ):
                        id_timestamp = id_timestamp_now + i
                        result.append( (
                            id_timestamp,
                            lat_now + i * slope_lat,
                            lng_now + i * slope_lng
                        ) )
                
                # Last TimeStamp
                id_timestamp_last, lat_last, lng_last = x_arr[0]
                for id_timestamp in range( id_timestamp_last, id_timestamp_max+1 ):
                    result.append( (id_timestamp, lat_last, lng_last) )
                
                # Return
                return( result )
            udf_linear_interpolator = sql_functions.udf(
                linear_interpolator,
                sql_types.ArrayType(
                    SCHEMA_DATA_POINT,
                    False
                )
            )
            # Aggregate, Apply UDF & Explode
            ## Aggregate Data Points
            df_result = df_trajectory_bucketed.withColumn(
                "data_point",
                udf_gather_data_point(
                    sql_functions.col("id_timestamp"),
                    sql_functions.col("lat"),
                    sql_functions.col("lng"),
                )
            ).groupby("id_user").agg(
                sql_functions.collect_list(
                    sql_functions.col("data_point")
                ).alias("data_point_list")
            )
            ## Apply Linear Interpolation
            df_result = df_result.select(
                "id_user",
                udf_linear_interpolator(
                    sql_functions.col("data_point_list")
                ).alias("data_point_list")
            )
            ## Explode to Comply with DataCatalogue
            df_result = df_result.withColumn(
                "data_point",
                sql_functions.explode( sql_functions.col("data_point_list") )
            ).select(
                "id_user",
                "data_point.id_timestamp",
                "data_point.lat",
                "data_point.lng"
            )
        # Return
        return( df_result )