from datetime import date
import geopandas as pd
from geopandas import GeoDataFrame

def getPollutantFromDateRange(conn, table, start_date, end_date, stations):
    """ Gets all the table names of our DB"""
    cur = conn.cursor();
    stations_str = "','".join(stations)
    print(stations_str)
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


def getAllStationsTxtGeom(conn, lastyear=date.today().year):
    """ Gets all the table names of our DB"""
    cur = conn.cursor();
    sql = F"""SELECT ST_AsText(ST_Transform(geom, 4326)) as geom, nombre 
                FROM cont_estaciones
                WHERE lastyear >= {lastyear}"""
    # print(sql)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return rows
