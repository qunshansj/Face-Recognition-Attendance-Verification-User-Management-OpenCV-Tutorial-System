
    def register(self):
        username = self.textEdit_1.toPlainText()
        password = self.textEdit_2.toPlainText()
        #print(username,password)


        # 执行的都是原生SQL语句
        sql = '''
        SELECT Name
        FROM   users
        '''

        has_name = 0
        for row in self.get(sql):
            if username == str(row[0]):
                #print(row)
                has_name = 1
                self.textEdit_1.setText('用户已经注册')
                break
        if not has_name:
            savepath = './images_db'
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            # 打开摄像头
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            cap = cv2.VideoCapture(0)
            for x in range(50):
                ret, image = cap.read()
                imagecopy = image.copy()
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
                faces = detector(img_gray, 0)

                # 待会要显示在屏幕上的字体
                font = cv2.FONT_HERSHEY_SIMPLEX

                # 如果检测到人脸
                if (len(faces) != 0):

                    for i in range(len(faces)):
                        # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                        for k, d in enumerate(faces):
                            # 用红色矩形框出人脸
                            cv2.rectangle(imagecopy, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0))
                            # 计算人脸热别框边长
                            face_width = d.right() - d.left()

                            shape = predictor(image, d)
                            mouth_width = (shape.part(54).x - shape.part(48).x) / face_width  # 嘴巴咧开程度
                            mouth_higth = (shape.part(66).y - shape.part(62).y) / face_width  # 嘴巴张开程度
                            for i in range(68):
                                if i > 27:
                                    cv2.circle(imagecopy, (shape.part(i).x, shape.part(i).y), 2, (0, 0, 255), -1, 2)

                self.showimg(imagecopy)
                QApplication.processEvents()
            cv2.imwrite(savepath + '/' + str(username) + '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            val = (str(username),str(password))
            sql = 'insert into users (Name , Password) values (%s, %s)'
            self.add(sql, val)
            self.textEdit_1.setText('注册成功')
