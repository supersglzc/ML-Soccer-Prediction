import logging
import tornado.auth
import tornado.escape
import tornado.ioloop
import tornado.web
import os.path
import uuid
import regression_tree
import simplejson as json

from tornado.concurrent import Future
from tornado import gen
from tornado.options import define, options, parse_command_line

define("port", default=8000, help="run on the given port", type=int)
define("debug", default=False, help="run in debug mode")

class BaseHandler(tornado.web.RequestHandler):
    def get_current_user(self):
        user_json = self.get_secure_cookie("user")
        if not user_json: return None
        return tornado.escape.json_decode(user_json)

class MainHandler(BaseHandler):
    def get(self):
        self.render("index.html")
    
class AdminHandler(BaseHandler):
    def get(self):
        self.render("admin.html")

class LoginAjaxHandler(tornado.web.RequestHandler):
    def post(self):
        usr=self.get_argument("name")
        pwd=self.get_argument("pwd")
        if usr=='123' and pwd=='123':
            self.finish('0')
        else:
            self.finish('-1')

class GetReqAjaxHandler(tornado.web.RequestHandler):
    def post(self):
        vType=self.get_argument("type")
        age=self.get_argument("age")
        hgt=self.get_argument("height")
        wht=self.get_argument("weight")
        tgt=self.get_argument("target")
        col_names=regression_tree.load_colName('./static/data/col_name.pkl')
        if vType=='Wage':
            tree=regression_tree.load_tree('./static/data/dataForWage.pkl')
        else:
            tree=regression_tree.load_tree('./static/data/dataForMarketValue.pkl')
        list1 = []
        list2 = ""
        regression_tree.find_route(tree, list1, list2, age, hgt, wht, col_names)
        requirement = regression_tree.target_route(float(tgt), list1)
        self.finish(json.dumps(requirement))

class GetQueAjaxHandler(tornado.web.RequestHandler):
    def post(self):
        col_names=regression_tree.load_colName('./static/data/col_name.pkl')
        self.finish(json.dumps(col_names))

class GetValAjaxHandler(tornado.web.RequestHandler):
    def post(self):
        vType=self.get_argument("type")
        if vType=='Wage':
            tree=regression_tree.load_tree('./static/data/dataForWage.pkl')
        else:
            tree=regression_tree.load_tree('./static/data/dataForMarketValue.pkl')
        inputs=self.get_argument("ans")
        inputs=inputs.split(',')[:-1]
        for i in range(len(inputs)):
            inputs[i]=float(inputs[i])
        target = regression_tree.calculate_value(regression_tree.classify(inputs, tree))
        if vType == 'Wage':
            result = str(target) + 'K'
        else:
            result = str(target) + 'M'
        self.finish(json.dumps(result))


class FileUploadHandler(tornado.web.RequestHandler):
    def post(self):
        ret = {'result': 'OK'}
        upload_path = './static/data/'

        file_metas = self.request.files.get('pkl_file', None)  

        if not file_metas:
            ret['result'] = 'Invalid Args'
            return ret
        
        for meta in file_metas:
            filename = meta['filename']
            file_path = os.path.join(upload_path, filename)

            with open(file_path, 'wb') as up:
                up.write(meta['body'])
 
        self.write(json.dumps(ret))


def main():
    parse_command_line()
    app = tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/login", LoginAjaxHandler),
            (r"/admin", AdminHandler),
            (r"/getReq", GetReqAjaxHandler),
            (r"/getQuestion",GetQueAjaxHandler),
            (r"/getValue",GetValAjaxHandler),
            (r'/data', FileUploadHandler),
        ],
        cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        xsrf_cookies=False,
        debug=options.debug,
    )
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
