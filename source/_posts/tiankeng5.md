---
title: python——flask后端代码开发
date: 2024-04-26 22:28:57
tags: 填坑
---
看别人的代码是必要的
## 背景介绍
最近我们在研究如何将直接的模型部署在网站上面，所以我们打算写一个python工程来放置对模型部署，使用的flask来进行模型和前端的交互。
大概思路是这样的，前端传入一个文件，经过flask传输到服务器，触发处理求取，模型处理完成后返回模型处理结果。
增加功能是用户可以自己选择模型种类和通道种类。（用户可以自由选取自己想处理的通道对应的模型）



## 挖坑

世界上没有技术驱动的公司，不论是google、facebook，还是阿里腾讯、阿里。技术不是源头，需求才是，因此一切技术问题，都要服从产品交付和市场反馈。所以，任何公司，都不可能以技术去驱动自身。人可以以技术驱动自己进步，但公司不行。以解决问题为导向，我需要解决什么问题，然后了解这个问题，有什么解决问题的思路。
资本富集的地方，人都得加班，加班的本质，是人跟着机器跑、人跟着钱跑；更为本质地说，资本富集的地方，人作为劳动力，也是资本的一种。即，人是资本而不是人本身。IT是工科，不是理科，和IT行业相似度最高的行业是盖楼房。真的，相似度相当惊人。
一个程序员，应该花80%的时间做代码设计、画UML图、画时序图，20%的时间写code和debug；菜鸟程序员的这个比例恰好是反的。一句话，不论这个需求有多紧急，你都一定要“想好再动手”；“想好”的标志就是设计文档写好了；文档一旦写好，写代码就是纯粹的无脑工作。
英语，很重要。能否使用英语查阅资料，是区分技术人员水平的重要指示之一。寄希望于“有人迟早会翻译成中文”的人是愚蠢的、是会被淘汰的。
工作要有热情。

智商决定你的起点情商决定你能走多远爬多高；混职场，靠的是情商。情商高就是：别人愿意和你一起工作、你有问题的时候别人愿意帮你。智商有时候可以稍微弥补一下情商但不起决定性的作用。


## Flask 
[参考博客](https://blog.csdn.net/wly55690/article/details/131683846)
1. Flask是一个非常小的PythonWeb框架，被称为微型框架；只提供了一个稳健的核心，其他功能全部是通过扩展实现的；意思就是我们可以根据项目的需要量身定制，也意味着我们需要学习各种扩展库的使用。
安装：
```
pip install flask
```
2. Flask 基础入门 
代码是1.py
1）路由route的创建：
```

# 通过创建路由并关联函数，实现一个基本的网页：
from flask import Flask

# 用当前脚本名称实例化Flask对象，方便flask从该脚本文件中获取需要的内容,app = Flask(__name__) 创建一个Flask实例
app = Flask(__name__)

#程序实例需要知道每个url请求所对应的运行代码是谁。
#所以程序中必须要创建一个url请求地址到python运行函数的一个映射。
#处理url和视图函数之间的关系的程序就是"路由"，在Flask中，路由是通过@app.route装饰器(以@开头)来表示的
@app.route("/")
#url映射的函数，要传参则在上述route（路由）中添加参数申明
def index():
    return "Hello World!"

# 直属的第一个作为视图函数被绑定，第二个就是普通函数
# 路由与视图函数需要一一对应
# def not():
#     return "Not Hello World!"

# 启动一个本地开发服务器，激活该网页
app.run()


```

- 通过路由的methods指定url允许的请求格式：
代码是2.py
```
# 访问方法 http://127.0.0.1:5000/hi
from flask import Flask

app = Flask(__name__)

#methods参数用于指定允许的请求格式
#常规输入url的访问就是get方法
@app.route("/hello",methods=['GET','POST'])
def hello():
    return "Hello World!"
#注意路由路径不要重名，映射的视图函数也不要重名 ,对于方法来讲，methods = []
@app.route("/hi",methods=['POST','GET'])
def hi():
    return "Hi World!"

# Ensure the app runs only if executed directly (not when imported as a module)
if __name__ == "__main__":
    app.run(debug=True)


```

代码是3.py
```
# 增加页面，
from flask import Flask

app = Flask(__name__)

# 可以在路径内以/<参数名>的形式指定参数，默认接收到的参数类型是string

'''#######################
以下为框架自带的转换器，可以置于参数前将接收的参数转化为对应类型
string 接受任何不包含斜杠的文本
int 接受正整数
float 接受正浮点数
path 接受包含斜杠的文本
########################'''

@app.route("/index/<int:id>",)
def index(id):
    if id == 1: # 访问方法：http://127.0.0.1:5000/index/1
        return 'first'
    elif id == 2: # 访问方法 http://127.0.0.1:5000/index/2
        return 'second'
    elif id == 3: # 访问方法 http://127.0.0.1:5000/index/3
        return 'thrid'
    else:
        return 'hello world!'

if __name__=='__main__':
    app.run()


```

- 除了原有的转换器，我们也可以自定义转换器（`pip install werkzeug`）：

```
from werkzeug.routing import BaseConverter #导入转换器的基类，用于继承方法
from flask import Flask

app = Flask(__name__)

# 自定义转换器类
class RegexConverter(BaseConverter):
    def __init__(self,url_map,regex):
        # 重写父类定义方法
        super(RegexConverter,self).__init__(url_map)
        self.regex = regex

    def to_python(self, value):
        # 重写父类方法，后续功能已经实现好了
        print('to_python方法被调用')
        return value

# 将自定义的转换器类添加到flask应用中
# 具体过程是添加到Flask类下url_map属性（一个Map类的实例）包含的转换器字典属性中
app.url_map.converters['re'] = RegexConverter
# 此处re后括号内的匹配语句，被自动传给我们定义的转换器中的regex属性
# value值会与该语句匹配，匹配成功则传达给url映射的视图函数
@app.route("/index/<re('1\d{10}'):value>")
def index(value):
    print(value)
    return "Hello World!"

if __name__=='__main__':
    app.run(debug=True)

```

### 跨域
http:// (协议)
http://hostname(主机名)：port（端口号） 这三个东西有一个不同就叫做跨域，跨域的本质是浏览器的安全保护。
怎么解决，后端允许跨域。