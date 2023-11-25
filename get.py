
    def get(self,sql):
        try:
            conn = MySQLdb.connect(
                host=host,  # 主机名
                user=user,  # 用户名
                passwd=passwd,  # 密码
                db=db)  # 数据库名

            # 查询前，必须先获取游标
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()  # 关闭游标连接
            conn.close()  # 关闭数据库连接
            return results
        except Exception:
            print("查询失败")

    def update(self,sql,cg):
        try:
            conn = MySQLdb.connect(
                host=host,  # 主机名
                user=user,  # 用户名
                passwd=passwd,  # 密码
                db=db)  # 数据库名

            # 查询前，必须先获取游标
            cursor = conn.cursor()
            cursor.execute(sql,cg)
            conn.commit()     # 提交
            cursor.close()  # 关闭游标连接
            conn.close()  # 关闭数据库连接

        except Exception:
            print("修改失败")

    def add(self,sql,val):
        try:
            conn = MySQLdb.connect(
                host=host,  # 主机名
                user=user,  # 用户名
                passwd=passwd,  # 密码
                db=db)  # 数据库名

            # 查询前，必须先获取游标
            cursor = conn.cursor()
            cursor.execute(sql, val)
            conn.commit()     # 提交
            cursor.close()  # 关闭游标连接
            conn.close()  # 关闭数据库连接
        except Exception:
            print("插入失败")

    def delete(self,sql):
        try:
            conn = MySQLdb.connect(
                host=host,  # 主机名
                user=user,  # 用户名
                passwd=passwd,  # 密码
                db=db)  # 数据库名

            # 查询前，必须先获取游标
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            cursor.close()  # 关闭游标连接
            conn.close()  # 关闭数据库连接
        except Exception:
            print("删除失败")
