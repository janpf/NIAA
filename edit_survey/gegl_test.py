import os
import gi

gi.require_version('Gegl', '0.4')
from gi.repository import Gegl
Gegl.init()

graph = Gegl.Node()
gegl_img = graph.create_child('gegl:load')
gegl_img.set_property('path', '/data/442284.jpg')

colorfilter = graph.create_child('gegl:shadows-highlights-correction')
colorfilter.set_property('shadows', 100)
gegl_img.link(colorfilter)

sink = graph.create_child('gegl:jpg-save')
sink.set_property('path', '/data/output.jpg')
colorfilter.link(sink)

sink.process()
