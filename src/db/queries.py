
def getPollutantFromDateRange(conn, table, start_date, end_date, stations):
    """ Gets all the table names of our DB"""
    cur = conn.cursor();
    sql = F""" SELECT fecha, val, id_est FROM {table} 
                WHERE fecha BETWEEN '{start_date}' AND '{end_date}'
                AND id_est IN ('{stations[0]}')
                ORDER BY fecha;"""
    # print(sql)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()

    return rows
