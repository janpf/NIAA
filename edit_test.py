from edit_image import *

# parameter generation test
# while True:
#    input(random_parameters())

change = "lcontrast"
value = "0"

edit_image("/data/442284.jpg", change, value, f"/data/result_{change}_{value}.jpg")
