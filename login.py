
    def login(self):
        global inname
        username = self.textEdit_1.toPlainText()
        password = self.textEdit_2.toPlainText()
        # print(username,password)

        # 执行的都是原生SQL语句
        sql = '''
                SELECT Name, Password
                FROM   users
                '''
        has_name = 0
        for row in self.get(sql):
            if username == 'administrator' and username == str(row[0]):
                has_name = 1
                if password == str(row[1]):
                    print('密码正确，登陆成功')
                    inname = str(row[0])
                    MainWindow.close()
                    ui4.show()
                    try:
                        show = cv2.imread('./images_db/' + str(inname) + '.jpg')
                        ui4.showimg(show)
                    except:
                        pass
                    ui4.printf('密码正确，登陆成功')
                    ui4.printf('欢迎您,管理员')
                else:
                    self.textEdit_2.setText('密码错误')
                break

            if username == str(row[0]):
                # print(row)
                has_name = 1
                if password == str(row[1]):
                    print('密码正确，登陆成功')
                    inname = str(row[0])
                    MainWindow.close()
                    ui2.show()
                    try:
                        show = cv2.imread('./images_db/' + str(inname) + '.jpg')
                        ui2.showimg(show)
                    except:
                        pass
                    ui2.printf('密码正确，登陆成功')
                    ui2.printf('欢迎您 ' + str(inname))
                else:
                    self.textEdit_2.setText('密码错误')
                break

        if not has_name:
            self.textEdit_1.setText('该用户不存在')
