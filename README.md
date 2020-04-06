# ![EastFlask](https://github.com/Fabriceli/EastFlask/blob/master/static/demo_home_page.png)
# EastFlask
A web server on Flask and with the model EAST, detection text where it is, simple demo. EAST model base on paper 
[EAST: An Efﬁcient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)


## Some Detection results
![](https://github.com/Fabriceli/EastFlask/blob/master/static/dect_img_5.jpg)

![](https://github.com/Fabriceli/EastFlask/blob/master/static/dect_img_165.jpg)

![](https://github.com/Fabriceli/EastFlask/blob/master/static/dect_OIP.jpeg)

![](https://github.com/Fabriceli/EastFlask/blob/master/static/dect_twitter.png)

## How to Use
1. [虚拟环境搭建](https://blog.csdn.net/YIQI521/article/details/105346104)

2. 安装虚拟环境依赖项
    ```bash
    pip install -r requirements.txt
    ```
3. 运行项目
    ```bash
    export FLASK_APP=app.py
    export FLASK_ENV=development
    flask run
    ```
    

MIT © [Fabrice LI](https://github.com/Fabriceli)
