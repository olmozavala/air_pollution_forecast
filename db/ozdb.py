class Ozdb:
    """
    This class contains a series of useful functions to manipulate databases
    """

    def __init__(self):
        """Constructor of the class"""

    def dropTable(self,table,conn):
        cur = conn.cursor()
        """Drops all the values of an specific table """
        print("Deleting table %s" % (table))
        sql = "DELETE FROM %s WHERE true;" % (table)
        cur.execute(sql)
        conn.commit()
        cur.close()

    def restartSeq(self,table,conn,cur):
        """Restarts a postgres sequence from a table to 1"""
        seq = "cont_seq_"+(table.replace("cont_",""))
        print("Setting sequence = 1 for seq %s" % (seq))
        sql = "ALTER SEQUENCE %s RESTART WITH 1" % (seq)
        print(sql)
        cur.execute(sql)
        conn.commit()
