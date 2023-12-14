from datetime import date
import geopandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame

def getPollutantFromDateRange(conn, table, start_date, end_date, stations):
    """ Gets all the values of a pollutant from a date range for a list of stations """
    cur = conn.cursor()
    stations_str = "','".join(stations)
    # print(stations_str)
    sql = F""" SELECT fecha, val, id_est FROM {table} 
                WHERE fecha BETWEEN '{start_date}' AND '{end_date}'
                AND id_est IN ('{stations_str}')
                ORDER BY fecha;"""
    # print(sql)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return rows

def getAllStations(conn, lastyear=date.today().year):
    """ Gets all the table names of our DB"""
    sql = F"""SELECT id, geom, nombre 
                FROM cont_estaciones
                WHERE lastyear >= {lastyear}"""
    return GeoDataFrame.from_postgis(sql, con=conn, index_col='id')


def getAllStationsNames(conn):
    cur = conn.cursor()
    """ Gets all the table names of our DB"""
    sql = F"""SELECT id, nombre 
                FROM cont_estaciones"""

    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    return rows

def getAllStationsTxtGeom(conn, lastyear=date.today().year):
    """ Gets all the table names of our DB"""
    cur = conn.cursor()
    sql = F"""SELECT ST_AsText(ST_Transform(geom, 4326)) as geom, nombre 
                FROM cont_estaciones
                WHERE lastyear >= {lastyear}"""
    # print(sql)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return rows

def getCountValidData(conn, year, sinceYear=False):
    """ Counts the number of valid data per station
    :param conn:
    :param year:
    :param station:
    :return:
    """
    cur = conn.cursor();
    if sinceYear:
        sql = F"""SELECT id_est, COUNT(*) FROM cont_otres 
                WHERE date_part('year',fecha) >= {year} and 
                date_part('year',fecha) <= {2017}
                    GROUP BY id_est"""
    else:
        sql = F"""SELECT id_est, COUNT(*) FROM cont_otres 
                WHERE date_part('year',fecha) = {year} GROUP BY id_est"""

    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return DataFrame(rows, columns=['id_est', 'value'])

def getMaxDailySQL(conn, year):
    cur = conn.cursor();
    sql = F""" SELECT max(val) as mval, 
                 date_part('day',fecha) as dia, 
                  date_part('month',fecha) as mes, 
                  date_part('year',fecha) as anio
              FROM cont_otres
              WHERE date_part('year',fecha) = {year}
              GROUP BY dia,mes,anio
              ORDER BY anio,mes,dia"""
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return DataFrame(rows, columns=['max', 'day', 'month', 'year'])
