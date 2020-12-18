import datetime
import dominate
from dominate.tags import *


class HTML:
    def __init__(self, web_dir, title, refresh=0):
        if web_dir.stem == '.html':
            web_dir, html_name = web_dir.parent, web_dir.name
        else:
            web_dir, html_name = web_dir, 'index.html'

        self.title = title
        self.web_dir = web_dir
        self.html_name = html_name
        self.img_dir = self.web_dir / 'images'
        if not self.web_dir.exists():
            # os.makedirs(self.web_dir)
            self.web_dir.mkdir(parents=True, exist_ok=True)
        if not self.img_dir.exists():
            self.img_dir.mkdir(parents=True, exist_ok=True)

        self.doc = dominate.document(title=title)
        with self.doc:
            h1(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=512):
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href='images/%s' % link):
                                img(style="width:%dpx" %
                                    (width), src='images/%s' % im)
                            br()
                            # p(txt.encode('utf-8')
                            p(txt)

    def save(self):
        html_file = self.web_dir / self.html_name
        with html_file.open(mode='wt') as f:
            f.write(self.doc.render())
            f.close()
